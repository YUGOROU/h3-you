"""Small source-only GPU-worker bundle. No weights, credentials, media, or jobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from research.install_identity_guidance import PINS,REVISION
from research.install_identity_checkpoint import plan

FILES=('research/hf_identity_encode.py','research/hf_identity_train.py','research/hf_identity_pipeline.py',
 'research/hf_prepare_identity_media.py','research/install_identity_guidance.py',
 'research/install_identity_component_cache.py','research/install_identity_sampling.py','research/install_identity_checkpoint.py',
 'research/h3_personal/freeze_dataset.py','research/h3_personal/production_inputs.py',
 'research/h3_personal/component_encode.py','research/h3_personal/component_cache.py',
 'research/h3_personal/identity_guidance.py','research/h3_personal/identity_sampling.py',
 'research/h3_personal/identity_checkpoint.py')


def build(project,source,out):
    project,source,out=map(Path,(project,source,out));plan(source)
    entries={name:project/name for name in FILES}
    entries.update({'upstream/'+name:source/name for name in PINS})
    entries['upstream/LICENSE']=source/'LICENSE'
    manifest={'schema':'h3-identity-source-bundle-v1','trainer_revision':REVISION,
              'files':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in entries.items()},
              'actual_h3_gpu_verified':False,'contains_weights':False,'contains_credentials':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for name,path in sorted(entries.items()):archive.add(path,arcname=name,recursive=False)
            data=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('CODE_MANIFEST.json');info.size=len(data);info.mode=0o600
            archive.addfile(info,io.BytesIO(data))
    return {'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'files':len(entries)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(Path(__file__).resolve().parents[1],a.source,a.output.expanduser()),indent=2))

if __name__=='__main__':main()
