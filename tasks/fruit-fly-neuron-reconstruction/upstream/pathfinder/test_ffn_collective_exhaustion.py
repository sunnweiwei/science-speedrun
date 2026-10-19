from pathlib import Path
import datetime,torch,unittest,tempfile
import torch.multiprocessing as mp

from train_volume import production_windows,loss_sum
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(.1))
 def forward(self,raw,canvas):return canvas+raw+self.bias

def compute(rank,stop,ddp,all_empty=False):
 model=Model();wrapped=torch.nn.parallel.DistributedDataParallel(model) if ddp else model
 raw=torch.full((1,1,65,65,65),6. if rank and not all_empty else -7.);mask=torch.ones_like(raw);calls=0
 for logits,y,m in production_windows(wrapped,raw,mask,mask,steps=4,stop_when_exhausted=stop):
  (loss_sum(logits,y,m)/m.sum().clamp_min(1)/4).backward();calls+=1
 return float(model.bias.grad),calls

def worker(rank,path):
 torch.set_num_threads(1);torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=30))
 for all_empty in (False,True):
  a=compute(rank,False,True,all_empty);b=compute(rank,True,True,all_empty)
  torch.testing.assert_close(torch.tensor(a[0]),torch.tensor(b[0]),rtol=1e-6,atol=1e-7)
  assert a[1]==4 and b[1]==(1 if all_empty else 4)
 torch.distributed.destroy_process_group()
class Tests(unittest.TestCase):
 def test_empty_steps_skipped_with_same_gradient(self):
  a=compute(0,False,False);b=compute(0,True,False)
  self.assertEqual(a[1],4);self.assertEqual(b[1],1);self.assertAlmostEqual(a[0],b[0],places=7)
 def test_mixed_rank_activity_preserves_ddp_and_gradient(self):
  with tempfile.TemporaryDirectory() as d:mp.spawn(worker,args=(str(Path(d)/'init'),),nprocs=2,join=True)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
