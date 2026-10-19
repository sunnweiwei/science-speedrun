"""Optional inference-only Welford LayerNorm for FFN32, following PyTorch 2.6 order.
Reference: github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu
Independent Triton implementation. Startup parity checks gate each model.
Measured runtime: PyTorch2.6.0, Triton3.2.0, A5000. No GPU-model gate.
"""
import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def combine(m1,v1,c1,m2,v2,c2):
 count=c1+c2;coef=tl.div_rn(1.,count);n1=c1*coef;n2=c2*coef;delta=m1-m2
 mean=tl.fma(n2,m2,n1*m1)
 var=tl.fma((delta*delta)*c2,n1,v2+v1)
 return mean,var,count

@triton.jit
def kernel(X,W,B,Y,M:tl.constexpr,EPS:tl.constexpr,BR:tl.constexpr,FMA:tl.constexpr):
 row=tl.program_id(0)*BR+tl.arange(0,BR);lane=tl.arange(0,8)
 active=row[:,None]<M
 mean=tl.full((BR,8),0,tl.float32);var=tl.full((BR,8),0,tl.float32)
 for j in tl.static_range(4):
  x=tl.load(X+row[:,None]*32+lane[None,:]*4+j,active,other=0)
  delta=x-mean
  if FMA==0:r=1.0/(j+1)
  else:r=tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",constraints="=f,f",args=[tl.full((BR,8),j+1,tl.float32)],dtype=tl.float32,is_pure=True,pack=1)
  newmean=tl.fma(delta,r,mean);var=tl.fma(delta,x-newmean,var);mean=newmean
 m,v,c=tl.reduce((mean,var,tl.full((BR,8),4,tl.float32)),1,combine)
 inv=tl.rsqrt(v*(1.0/32)+EPS)
 for j in tl.static_range(4):
  ch=lane*4+j;x=tl.load(X+row[:,None]*32+ch[None,:],active,other=0);w=tl.load(W+ch);b=tl.load(B+ch)
  y=tl.fma(w[None,:],inv[:,None]*(x-m[:,None]),b[None,:]);tl.store(Y+row[:,None]*32+ch[None,:],y,active)

class Norm32(torch.nn.Module):
 def __init__(self,original,fma=0,br=4):
  super().__init__();self.weight=original.weight;self.bias=original.bias;self.eps=original.eps;self.fma=fma;self.br=br;self.train(original.training)
 def forward(self,x):
  if self.training or torch.is_grad_enabled():raise ValueError('Norm32 is inference-only')
  if x.shape[-1]!=32:raise ValueError('Norm32 requires exactly32 channels')
  x=x.float().contiguous();out=torch.empty_like(x);rows=x.numel()//32
  kernel[(triton.cdiv(rows,self.br),)](x,self.weight,self.bias,out,rows,self.eps,self.br,self.fma,num_warps=4,enable_fp_fusion=False)
  return out


def install_norm32(model):
 """Validate supported runtime and all layer outputs before replacing any norm."""
 if model.training:raise ValueError('An eval-mode FFN model is required')
 device=next(model.parameters()).device
 if device.type!='cuda':raise ValueError('norm32 requires CUDA')
 if str(torch.__version__).split('+')[0]!='2.6.0' or triton.__version__!='3.2.0':
  raise ValueError('norm32 requires validated PyTorch2.6.0 and Triton3.2.0')
 replacements=[]
 generator=torch.Generator(device=device).manual_seed(8173)
 with torch.inference_mode():
  probe=torch.randn((33,33,33,32),device=device,generator=generator)
  for block in model.blocks:
   norm=block.norm
   if not isinstance(norm,torch.nn.LayerNorm) or tuple(norm.normalized_shape)!=(32,) or norm.weight.dtype!=torch.float32 or norm.bias is None:
    raise ValueError('Expected float32 affine32-channel LayerNorm')
   fast=Norm32(norm,fma=1,br=4)
   if not torch.equal(norm(probe),fast(probe)):
    raise ValueError('LayerNorm startup parity failed; use eager inference')
   replacements.append((block,fast))
 for block,fast in replacements:block.norm=fast


class CudaGraphFFN:
 """Exact-shape CUDA graph cache for independent full33 FFN requests.

 No padding, changed batch sizes, altered weights, or inference-time training.
 CPU output is copied before replay so callers retain independent arrays.
 """
 def __init__(self,model,max_batch,bf16=True):
  if type(max_batch) is not int or max_batch<1:raise ValueError('Positive max_batch required')
  install_norm32(model)
  self.model=model;self.max_batch=max_batch;self.bf16=bf16
  self.device=next(model.parameters()).device;self.cache={}
 def _capture(self,images,canvases):
  batch=len(images);a=torch.empty((batch,1,33,33,33),device=self.device);b=torch.empty_like(a)
  a.copy_(torch.from_numpy(images[:,None]));b.copy_(torch.from_numpy(canvases[:,None]))
  stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream(self.device))
  # Warmup casts must not escape through AMP's cache into the captured graph:
  # those temporary weights are freed when the autocast context exits.
  # Capture the casts themselves so their storage belongs to the graph pool.
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16,cache_enabled=False):
   with torch.cuda.stream(stream):
    for _ in range(4):output=self.model(a,b)[:,0].float()
   torch.cuda.current_stream(self.device).wait_stream(stream)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):output=self.model(a,b)[:,0].float()
  self.cache[batch]=(a,b,output,graph)
 def __call__(self,images,canvases):
  if (not isinstance(images,np.ndarray) or not isinstance(canvases,np.ndarray)
      or images.dtype!=np.float32 or canvases.dtype!=np.float32 or images.shape!=canvases.shape
      or images.ndim!=4 or images.shape[1:]!=(33,33,33) or not 1<=len(images)<=self.max_batch):
   raise ValueError('Matching float32 B,33,33,33 inputs within max_batch required')
  batch=len(images)
  if batch not in self.cache:self._capture(images,canvases)
  a,b,output,graph=self.cache[batch]
  with torch.inference_mode():
   a.copy_(torch.from_numpy(images[:,None]));b.copy_(torch.from_numpy(canvases[:,None]));graph.replay()
   return output.cpu().numpy()
