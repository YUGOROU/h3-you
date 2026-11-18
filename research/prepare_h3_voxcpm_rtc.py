"""Explicit RTC-only Vox Endpoint variant; staging only, never cloud activation."""
import hashlib
import shutil
from pathlib import Path
from research.run_h3_native_coupled import copy_runtime,ROOT,persist,deployment as legacy_deployment


def stage(root,spec):
    root=Path(root);copy_runtime(root,spec)
    from research.launch_h3_direct_conversation import first_party_sources
    for source in first_party_sources(ROOT):
        target=root/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    shutil.copyfile(ROOT/'research/endpoints/voxcpm_l4/entrypoint_rtc.py',root/'entrypoint_rtc.py')
    seal={'schema':'h3-vox-rtc-runtime-v1'}
    for name,key in (('pyproject.toml','project_sha256'),('uv.lock','lock_sha256')):
        source=ROOT/'research/webrtc_experiment'/name
        target=root/'research/webrtc_experiment'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);seal[key]=hashlib.sha256(target.read_bytes()).hexdigest()
    persist(root/'vox-rtc.json',seal)
    return root


def deployment(repo,revision,name,session,boot,expires):
    row=legacy_deployment(repo,revision,name,session,boot,expires)
    row['model']['command']=['python','/repository/entrypoint_rtc.py','/repository']
    row['model']['env']['H3_VOX_MEDIA_TRANSPORT']='webrtc-exact-v1'
    row['tags']=['h3-native-voxcpm-l4','finite-rtc-native-v3']
    return row
