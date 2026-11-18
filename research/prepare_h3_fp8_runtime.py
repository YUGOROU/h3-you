"""Pinned additive TorchAO install in a restored experimental Job venv only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from research.h3_personal.acceleration_experiment import WHEEL_NAME,WHEEL_SHA,WHEEL_URL,TORCHAO_VERSION


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--directory',type=Path,default=Path('/work/fp8-dependency'));a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(plan_only=True,version=TORCHAO_VERSION,wheel_sha256=WHEEL_SHA,
                             torch_upgrade=False,gpu_submitted=False)));return 0
    import torch
    if (not Path(sys.prefix).resolve().is_relative_to(Path('/work/runtime').resolve()) or
            torch.__version__!='2.8.0+cu128' or torch.cuda.device_count()!=2 or
            any('H200' not in torch.cuda.get_device_name(i) for i in range(2))):
        raise ValueError('install restricted to restored isolated twoH200 Job runtime')
    before={n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers','peft')}
    a.directory.mkdir(mode=0o700);wheel=a.directory/WHEEL_NAME
    with urllib.request.urlopen(WHEEL_URL,timeout=60) as source,wheel.open('xb') as target:
        total=0
        while chunk:=source.read(1024*1024):
            total+=len(chunk)
            if total>256*1024**2:raise ValueError('bounded wheel download required')
            target.write(chunk)
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=WHEEL_SHA:raise ValueError('TorchAO wheel SHA differs')
    with zipfile.ZipFile(wheel) as archive:
        for n in archive.namelist():
            path=Path(n)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('torchao','torchao.libs','torchao-0.13.0.dist-info'):
                raise ValueError('wheel may not overwrite base dependency namespaces')
    subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps','--no-index',str(wheel)],check=True,timeout=180)
    if before!={n:importlib.metadata.version(n) for n in before}:raise ValueError('base package version changed')
    if importlib.metadata.version('torchao')!=TORCHAO_VERSION:raise ValueError('TorchAO install version differs')
    # Fail fast on tiny real kernels BEFORE model inventory/conditioning/reload.
    from torchao.quantization import quantize_
    from torchao.quantization.quantize_.workflows import Float8Tensor
    from research.h3_personal.acceleration_experiment import fp8_recipe,probe_fp8_linear
    probes=[]
    for device in range(2):
        with torch.cuda.device(device):
            for fast_accum in (False,True):
                sample=torch.nn.Linear(256,256,device=f'cuda:{device}',dtype=torch.bfloat16).eval().requires_grad_(False)
                quantize_(sample,fp8_recipe(fast_accum))
                if not isinstance(sample.weight,Float8Tensor):raise ValueError('preflight FP8 conversion skipped')
                result=dict(device=device,fast_accum=fast_accum,**probe_fp8_linear(sample))
                probes.append(result);print('FP8_KERNEL_PREFLIGHT',json.dumps(result),flush=True)
                del sample
            torch.cuda.empty_cache()
    (a.directory/'receipt.json').write_text(json.dumps(dict(wheel_sha256=WHEEL_SHA,base_versions=before,
        installed=TORCHAO_VERSION,torch_upgrade=False,activation_value_lb=1e-12,
        kernel_preference='TORCH',kernel_probes=probes,quality_verified=False),indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
