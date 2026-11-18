"""Offline ASR RTC variant staging; native model/runtime artifacts unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
from research.launch_h3_conversation_scenario import closure
from research.endpoints.nemotron_t4.entrypoint_rtc import RUNTIME_SHA,MODEL_SHA

ROOT=Path(__file__).resolve().parents[1]


def stage(directory):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):raise ValueError('empty ASR source stage required')
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'research/endpoints/nemotron_t4/entrypoint_rtc.py',directory/'entrypoint.py')
    # Preserve explicit adapter pin naming without including a public WS gate.
    (directory/'gate.py').write_text('"""RTC-only ASR control gate; no public WebSocket audio handler."""\nfrom research.h3_personal.asr_rtc_gateway import Server\n')
    closure(directory,['research.run_h3_asr_rtc_service','research.h3_personal.webrtc_sidecar'])
    for name in ('pyproject.toml','uv.lock'):
        source=ROOT/'research/webrtc_experiment'/name
        if source.is_symlink():raise ValueError('isolated RTC lock symlink')
        target=directory/'research/webrtc_experiment'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/'asr-rtc.json').write_text(json.dumps(dict(schema='h3-asr-rtc-runtime-v1',runtime_sha256=RUNTIME_SHA,model_sha256=MODEL_SHA,
        project_sha256=digest(directory/'research/webrtc_experiment/pyproject.toml'),lock_sha256=digest(directory/'research/webrtc_experiment/uv.lock')),sort_keys=True))
    manifest={str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    (directory/'rtc-source-manifest.json').write_text(json.dumps(manifest,sort_keys=True))
    return manifest
