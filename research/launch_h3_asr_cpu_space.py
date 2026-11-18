"""Explicit private CPU Basic Space probe. Offline staging by default.
No GPU, paid upgrade, personal audio upload, capture, auto-renewal or secrets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

SOURCE = Path(__file__).resolve().parent/'spaces/nemotron_cpu_basic'
FILES = ('Dockerfile', 'README.md', 'start.sh')


def stage(directory):
    directory = Path(directory)
    target = directory/'deployment'
    plan_path = directory/'plan.json'
    if not plan_path.exists():
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        target.mkdir()
        for name in FILES:
            shutil.copyfile(SOURCE/name, target/name)
        plan = dict(schema='h3-private-cpu-asr-space-v1', repo_id='YUGOROU/h3-nemotron-cpu-basic-'+uuid.uuid4().hex[:10],
                    private=True, sdk='docker', hardware='cpu-basic', expires_at=int(time.time())+2700,
                    max_server_seconds=1800, hourly_price_usd='0', public_samples_only=True,
                    files={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in FILES})
        plan_path.write_text(json.dumps(plan, indent=2))
    plan = json.loads(plan_path.read_text())
    if plan['private'] is not True or plan['hardware'] != 'cpu-basic' or plan['sdk'] != 'docker' or plan['public_samples_only'] is not True:
        raise ValueError('private CPU Basic public-sample-only contract required')
    if set(plan['files']) != set(FILES) or sorted(p.name for p in target.iterdir()) != sorted(FILES):
        raise ValueError('exact upload allowlist required')
    for name in FILES:
        if hashlib.sha256((target/name).read_bytes()).hexdigest() != plan['files'][name]:
            raise ValueError('immutable deployment changed')
    return plan, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--deploy', action='store_true')
    actions.add_argument('--status', action='store_true')
    actions.add_argument('--pause', action='store_true')
    args = p.parse_args()
    plan, target = stage(args.directory)
    if not (args.deploy or args.status or args.pause):
        print(json.dumps(plan))
        return
    from huggingface_hub import HfApi
    from research.provision_modal_identity_secret import cached_token
    api = HfApi(token=cached_token())
    repo = plan['repo_id']
    receipt = args.directory/'submission.json'
    if args.deploy:
        if receipt.exists() or plan['expires_at']-time.time() < 900:
            raise ValueError('do not replay deployment or deploy near expiry')
        created = False
        try:
            api.create_repo(repo, repo_type='space', private=True, space_sdk='docker', space_hardware='cpu-basic',
                            exist_ok=False, space_variables=[dict(key='H3_ASR_EXPIRES_AT', value=str(plan['expires_at']))])
            created = True
            if not api.space_info(repo).private:
                raise ValueError('private Space verification failed')
            commit = api.upload_folder(repo_id=repo, repo_type='space', folder_path=target,
                                       allow_patterns=list(FILES), commit_message='Finite private CPU Basic Nemotron public-sample streaming probe')
            value = dict(repo_id=repo, revision=commit.oid, private=True, hardware='cpu-basic',
                         expires_at=plan['expires_at'], hourly_price_usd='0', deployed_at=time.time())
            with receipt.open('x') as f:
                json.dump(value, f, indent=2)
            print(json.dumps(value))
        except BaseException:
            if created:
                api.pause_space(repo)
            raise
        return
    if not receipt.exists():
        raise ValueError('owned deployed Space receipt required')
    info = api.space_info(repo)
    if not info.private:
        raise ValueError('private Space boundary lost; no inference permitted')
    if args.pause:
        runtime = api.pause_space(repo)
    else:
        runtime = api.get_space_runtime(repo)
    hardware = str(runtime.hardware) if runtime.hardware else None
    if hardware is not None and hardware != 'cpu-basic':
        raise ValueError('unexpected hardware; no implicit paid upgrade')
    out = dict(repo_id=repo, stage=str(runtime.stage), hardware=hardware,
               requested_hardware=str(runtime.requested_hardware) if runtime.requested_hardware else None,
               host=getattr(info,'host',None), expires_at=plan['expires_at'])
    (args.directory/'status.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
