"""Fetch pinned Hub metadata only; no weights, uploads, allocations or submission."""
import argparse
import json
from pathlib import Path
from research.h3_personal.ref2va_runtime import BASE_REPO,BASE_REV,validate_model_manifest
from research.prepare_h3_realtime_first import metadata_entry,write_json


def manifest(api):
    info=api.model_info(BASE_REPO,revision=BASE_REV,files_metadata=True)
    if info.sha!=BASE_REV:raise ValueError('base revision differs')
    directories={'transformer_ref','text_encoder','tokenizer','processor','vae','audio_vae','scheduler','audio_scheduler'}
    files=[metadata_entry(s) for s in info.siblings if s.rfilename.split('/')[0] in directories
        or s.rfilename in ('model_index.json','modular_model_index.json','LICENSE','README.md')]
    result=dict(repo=BASE_REPO,revision=BASE_REV,files=files)
    validate_model_manifest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.output.exists() or a.output.resolve().is_relative_to(root):raise ValueError('new private output path required')
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    result=manifest(HfApi(token=cached_token()));write_json(a.output,result)
    print(json.dumps(dict(files=len(result['files']),bytes=sum(r['bytes'] for r in result['files']),
        weights_downloaded=False,gpu_allocated=False)))

if __name__=='__main__':main()
