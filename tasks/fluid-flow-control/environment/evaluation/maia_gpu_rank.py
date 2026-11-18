"""One explicit CUDA device per native MAIA solver rank (Python is rank zero)."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def cuda_uuids():
    driver=ctypes.CDLL('libcuda.so.1')
    def check(code):
        if code:
            raise RuntimeError(f'CUDA device enumeration failed: {code}')
    check(driver.cuInit(0))
    count=ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    result=[]
    for index in range(count.value):
        device=ctypes.c_int()
        raw=(ctypes.c_ubyte*16)()
        check(driver.cuDeviceGet(ctypes.byref(device),index))
        check(driver.cuDeviceGetUuid(ctypes.byref(raw),device))
        result.append('GPU-'+str(uuid.UUID(bytes=bytes(raw))))
    return result


SHARED_GPU_ENV = 'HYDROGYM_ALLOW_SHARED_GPU'


def allow_shared_gpu(allow_shared=None):
    if allow_shared is not None:
        return bool(allow_shared)
    value = os.environ.get(SHARED_GPU_ENV, '1').strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid {SHARED_GPU_ENV}={value!r}; expected 0 or 1')


def select_device(rank, devices, allow_shared=None):
    if type(rank) is not int or rank < 1 or not devices or len(set(devices)) != len(devices):
        raise ValueError('Need one distinct visible GPU for every MAIA solver rank')
    if rank <= len(devices):
        return devices[rank - 1]
    if allow_shared_gpu(allow_shared):
        return devices[(rank - 1) % len(devices)]
    raise ValueError('Need one distinct visible GPU for every MAIA solver rank')


def main():
    if sys.argv[1:]==['--list-gpus']:
        print(json.dumps(cuda_uuids()))
        return
    rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
    if rank==0 and sys.argv[1:]==['--binding-check']:
        return
    # A fresh child reads the caller's actual CUDA visibility/order. Do not
    # replace a caller's device mask with all GPUs listed by NVML/nvidia-smi.
    devices=json.loads(subprocess.check_output(
        [sys.executable,str(Path(__file__).resolve()),'--list-gpus'],text=True))
    selected=select_device(rank,devices)
    os.environ.update(CUDA_VISIBLE_DEVICES=selected,ACC_DEVICE_NUM='0',NV_ACC_DEVICE_NUM='0')
    if cuda_uuids()!=[selected]:
        raise RuntimeError('CUDA rank isolation did not take effect')
    record={'global_rank':rank,'visible_cuda_devices':1,'cuda_device_uuid':selected,
            'visible_pool_size':len(devices),'shared_gpu_mode':rank>len(devices),
            'container_pid':os.getpid()}
    output=Path(os.environ.get('HYDROGYM_GPU_BINDING_DIR','.'))
    (output/f'gpu-binding-rank-{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
    print('VERIFIED_MAIA_GPU_BINDING',json.dumps(record),flush=True)
    if sys.argv[1:]==['--binding-check']:
        return
    os.execvp('maia',['maia',*sys.argv[1:]])


if __name__=='__main__':
    main()
