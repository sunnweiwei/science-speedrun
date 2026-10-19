"""Build only the pinned PointNeXt FPS kernel, with no GPU execution.

Source and upstream MIT license remain under the budget directory/vendor.
"""
import argparse
import os
import shutil
from pathlib import Path
if 'CUDA_HOME' not in os.environ:
    compiler=shutil.which('nvcc')
    if compiler is None:
        raise RuntimeError('Set CUDA_HOME to an installed toolkit or put nvcc on PATH')
    os.environ['CUDA_HOME']=str(Path(compiler).resolve().parents[1])
from torch.utils.cpp_extension import load


def build(root):
    root=Path(root);src=root/'vendor/pointnext/openpoints/cpp/pointnet2_batch/src'
    directory=root/'point-ops';directory.mkdir(exist_ok=True)
    binding=directory/'binding.cpp'
    binding.write_text('#include <torch/extension.h>\n#include "sampling_gpu.h"\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("fps", &furthest_point_sampling_wrapper);}\n')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6');os.environ.setdefault('MAX_JOBS','2')
    return load(name='pathfinder_fps',sources=[str(binding),str(src/'sampling.cpp'),str(src/'sampling_gpu.cu')],
        extra_include_paths=[str(src)],extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],build_directory=str(directory),verbose=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
