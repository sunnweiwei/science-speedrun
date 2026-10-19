"""Record isolation evidence inside the CPU scoring container."""
import argparse,json,os,platform
from pathlib import Path
import numpy as np
import networkx as nx

p=argparse.ArgumentParser();p.add_argument('--host-netns',required=True);p.add_argument('--forbidden',nargs='+',required=True);a=p.parse_args()
visible=[name for name in a.forbidden if Path(name).exists()]
namespace=os.readlink('/proc/self/ns/net')
if visible:raise RuntimeError('Host paths remain visible: '+str(visible))
if namespace==a.host_netns:raise RuntimeError('Network namespace was not isolated')
if Path('/dev/nvidia0').exists():raise RuntimeError('GPU device unexpectedly visible')
mounts=Path('/proc/mounts').read_text().splitlines()
for target in ('/code','/prediction','/trusted'):
    matches=[line.split() for line in mounts if line.split()[1]==target]
    if not matches or any('ro' not in row[3].split(',') for row in matches):raise RuntimeError('Readonly bind not verified: '+target)
Path('/output/isolation.json').write_text(json.dumps(dict(host_paths_hidden=a.forbidden,network_namespace=namespace,
    host_network_namespace=a.host_netns,read_only_mounts=['/code','/prediction','/trusted'],gpu_visible=False,
    python=platform.python_version(),numpy=np.__version__,networkx=nx.__version__),indent=2)+'\n')
