"""Explicit finite resident live-session launcher; offline prepare/inspect by default."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from research.launch_h3_fast_ref2va import inspect as base_inspect,pre_submit as base_pre_submit
from research.launch_h3_realtime_first import main as launch
from research.prepare_h3_fast_ref2va_gpu import prepare
from research.h3_personal.ref2va_budget import budget_bound
from research.h3_personal.director_tts_job import REFERENCE_REPO,REFERENCE_REV
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()
ASR_POLICIES=('apple-final','application-stream-v1','endpoint-native-v1')
EXTRA=('research/h3_personal/streaming_asr.py','research/hf_h3_live_voice.py','research/h3_personal/live_voice_runtime.py','research/h3_personal/director_irodori.py','research/h3_personal/director_tts_job.py','research/hf_irodori_tts_diagnose.py',
       'research/h3_personal/continuation_plan.py','research/h3_personal/fast_continuation.py',
       'research/h3_personal/fast_continuation_stages.py','research/h3_personal/fast_continuation_decode.py',
       'research/h3_personal/fast_continuation_overlay.py','research/h3_personal/fast_continuation_render.py',
       'research/h3_personal/incremental_decode.py')


def inspect(stage):
    spec,inventory=base_inspect(stage);live=spec.get('live_voice_session')
    direct=spec.get('live_direct_session')
    long_session=type(direct) is dict and direct.get('long_session') is True
    if long_session:
        from research.hf_h3_direct_conversation import validate_spec as validate_direct_spec
        validate_direct_spec(spec)
    if spec.get('asr_policy','apple-final') not in ASR_POLICIES:raise ValueError('unsupported ASR policy')
    if not isinstance(live,dict) or set(live)!={'id','max_turns','fresh_input_required'} or live['max_turns'] not in ((1000,) if long_session else (2,16)) or live['fresh_input_required'] is not True or len(live['id'])!=32:raise ValueError('bounded fresh live session required')
    allowed_timeouts=(10800,) if long_session else (660,1200,1500)
    if spec['config']['timeout_seconds'] not in allowed_timeouts or spec.get('locked_speech') is not None or spec['layout']!='portrait' or spec['runtime_policy']!='resident-visual-cached' or spec['attention_mode']!='fa4-packed' or spec.get('regional_compile') or spec.get('face_crop_box') is not None:raise ValueError('resident v12 policy differs')
    if spec.get('asr_policy')=='endpoint-native-v1' and spec['config']['timeout_seconds'] not in allowed_timeouts:raise ValueError('native Endpoint requires finite660/1200/1500s HF H200x2')
    rows={x['path'] for x in inventory['files']}
    direct=spec.get('live_direct_session')
    if type(direct) is dict and direct.get('tts_mode')=='native_voxcpm':
        from research.hf_h3_direct_conversation import validate_spec
        validate_spec(spec)
        if not set(EXTRA)<=rows or 'tts-manifest.json' in rows:
            raise ValueError('native remote runtime must omit local TTS manifest')
        return spec,inventory
    if not set(EXTRA)|{'tts-manifest.json'}<=rows:raise ValueError('resident runtime missing')
    expected=json.loads((PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json').read_text())
    selected={k:expected[k] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}
    if json.loads((stage/'tts-manifest.json').read_text())!=selected:raise ValueError('native TTS manifest differs')
    return spec,inventory


def pre_submit(api,spec,stage):
    direct=spec.get('live_direct_session')
    native=type(direct) is dict and direct.get('tts_mode')=='native_voxcpm'
    if native:
        from research.hf_h3_direct_conversation import validate_spec
        validate_spec(spec)
        from research.h3_personal.native_tts_backend import validate_native_lease
        validate_native_lease(direct['native_tts'],min_remaining=120)
        if (Path(stage)/'tts-manifest.json').exists():
            raise ValueError('native remote backend forbids local TTS manifest')
    base_pre_submit(api,spec,stage)
    if native:
        return  # remote lease owns voice/model proof; no Irodori model checks
    api.auth_check(REFERENCE_REPO,repo_type='dataset',write=False)
    if not api.repo_info(REFERENCE_REPO,repo_type='dataset',revision=REFERENCE_REV).private:raise ValueError('private consenting speaker required')
    for m in json.loads((stage/'tts-manifest.json').read_text())['models']:
        api.auth_check(m['repo'],write=False)
        if api.model_info(m['repo'],revision=m['revision']).sha!=m['revision']:raise ValueError('native TTS revision unavailable')
        if m['repo']=='sony/silentcipher' and api.model_info(m['repo'],revision='main').sha!=m['revision']:raise ValueError('watermark main drift')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--submit',action='store_true');p.add_argument('--asr-mode',choices=ASR_POLICIES,default='apple-final');p.add_argument('--session-seconds',type=int,choices=(660,1200,1500),default=660);a=p.parse_args();stage=a.directory/'deployment'
    if not stage.exists():
        from huggingface_hub import HfApi
        from research.provision_modal_identity_secret import cached_token
        a.directory.mkdir(mode=0o700)
        prepare(stage,PRIVATE/'gpu-hf-realtime-personal-v1/deployment',PRIVATE/'compute-budget.json',HfApi(token=cached_token()),publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-live-'+uuid.uuid4().hex[:10],timeout_seconds=a.session_seconds,attention_mode='fa4-packed',layout='portrait',runtime_policy='resident-visual-cached')
        old=PRIVATE/'gpu-hf-irodori-diagnostic-v1/request.json'
        if hashlib.sha256(old.read_bytes()).hexdigest()!='b1279cf430f00314f8ac6f61728341fb174ad765fd4ea3bd5528c43c80800002':raise ValueError('original manifest drift')
        source=json.loads(old.read_text());manifest={k:source[k] for k in ('source_revision','uv_lock_sha256','uv_version','models','license')}
        (stage/'tts-manifest.json').write_text(json.dumps(manifest))
        for name in EXTRA:
            dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        spec=json.loads((stage/'spec.json').read_text());spec['live_voice_session']=dict(id=uuid.uuid4().hex,max_turns=16,fresh_input_required=True)
        spec['asr_policy']=a.asr_mode
        (stage/'spec.json').write_text(json.dumps(spec,indent=2))
        rows=[dict(path=q.relative_to(stage).as_posix(),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(stage.rglob('*')) if q.is_file() and q.name!='upload-allowlist.json']
        (stage/'upload-allowlist.json').write_text(json.dumps(dict(files=rows)))
    spec,_=inspect(stage)
    if a.submit and spec.get('asr_policy','apple-final')!=a.asr_mode:raise ValueError('requested ASR mode differs from immutable staged policy')
    args=['--stage',str(stage),'--ledger',str(PRIVATE/'compute-budget.json')]+(['--submit'] if a.submit else [])
    return launch(args,inspect_fn=inspect,budget_fn=budget_bound,pre_submit=pre_submit,bootstrap_entry='research/bootstrap_h3_fast_ref2va.py',approval_request='Current USER explicitly selects HF for H3+Irodori, superseding Modal-only restriction, and approves live Discord with echo prevention. Latest USER increases remaining AUTHORIZED spend to9.40USD at ledger bound221.672274; cumulative cap231.072274. This REPLACES remaining2.613326, not9.40 added on top and not inferred from provider credit. Fresh full-pipeline reservation checks include H3 1.833337 + ASR .75 + Pi .20. NEW live conversation input after resident ready only, no historical speech replay. USER approves resident multi-turn asynchronous session, no per-turn Pause/cancel; one finite660/1200/1500s HF H200x2 allocation for endpoint-native-v1; session-end/error/budget cleanup remains; original64/v12/BF16SMD4 and nativeFP32Irodori unchanged. Endpoint ASR is remote, not on-device; no raw audio uploaded to live session repo. Existing OBS only, no automatic routing/camera/join changes. No automatic extension/provider fallback.')

if __name__=='__main__':main()
