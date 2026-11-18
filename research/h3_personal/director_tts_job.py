"""Single fresh Pi-text → Irodori audio Job contract; no H3/device/fallback."""
import hashlib
import json
from pathlib import Path,PurePosixPath
from research.h3_personal.director_irodori import SOURCE,MODEL,CODEC

IMAGE='pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385'
REFERENCE_REPO='YUGOROU/h3-irodori-japanese-diagnostic-20261014-v1'
REFERENCE_REV='fadd78ca615a06949fcad8ebab489b166d6fa6f1'
REFERENCE_SHA='def3cd0487dff827a09e99498c13cd1e07daee741ec258935763efe2dae0fad1'
LOCK_SHA='8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c'
SCOPE='h3-director-new-irodori-turn-v1'
FILES={'request.json','research/hf_irodori_turn.py','research/hf_irodori_tts_diagnose.py','research/h3_personal/director_irodori.py','research/h3_personal/director_tts_job.py'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DEFAULT_TIMEOUT_SECONDS=1200
MAX_TIMEOUT_SECONDS=1200
INPUT_ORIGINS=('user_text_chat_not_discord_asr','discord_received_audio_ondevice_asr')


def validate_options(timeout_seconds,input_origin):
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=MAX_TIMEOUT_SECONDS:raise ValueError('timeout_seconds must be an integer in 1..1200')
    if input_origin not in INPUT_ORIGINS:raise ValueError('director/origin binding differs')


def validate(request):
    validate_options(request.get('timeout_seconds'),request.get('input_origin'))
    text=request.get('text')
    if not isinstance(text,str) or not text.strip() or len(text)>160:raise ValueError('one bounded fresh reply required')
    if request.get('director_model')!='Qwen/Qwen3.8-27B:cerebras':raise ValueError('director/origin binding differs')
    if request.get('scope')!=SCOPE or request.get('max_syntheses')!=1:raise ValueError('single finite audio turn only')
    if request.get('reference_sha256')!=REFERENCE_SHA or request.get('source_revision')!=SOURCE or request.get('uv_lock_sha256')!=LOCK_SHA or request.get('uv_version')!='0.13.1':raise ValueError('source/reference/lock drift')
    if request.get('user_approved_private_reference_upload') is not True:raise ValueError('voice consent binding required')
    models={m['repo']:m for m in request['models']}
    pins={'Aratako/Irodori-TTS-v4-Large':MODEL,'Aratako/Semantic-DACVAE-Japanese-32dim':CODEC,'sony/silentcipher':'a1c4d021905e0dc5b24be5f68db5fc4dba410ee1'}
    if len(models)!=3 or len(request['models'])!=3 or set(models)!=set(pins) or any(models[k]['revision']!=v for k,v in pins.items()):raise ValueError('model inventory drift')
    if not request['output_repo'].startswith('YUGOROU/h3-director-irodori-turn-'):raise ValueError('wrong private output scope')
    if not request.get('license'):raise ValueError('Gemma/voice approval binding missing')
    return request


def inspect(stage):
    stage=Path(stage);inventory=json.loads((stage/'upload-allowlist.json').read_text())
    rows={r['path']:r for r in inventory['files']}
    if len(rows)!=len(inventory['files']) or set(rows)!=FILES:raise ValueError('deployment inventory differs')
    if any(p.is_symlink() for p in stage.rglob('*')):raise ValueError('no symlink deployment')
    if {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}!=FILES|{'upload-allowlist.json'}:raise ValueError('unreviewed files')
    for name,row in rows.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe deployment path')
        local=stage/name
        if local.stat().st_size!=row['bytes'] or digest(local)!=row['sha256']:raise ValueError('deployment hash differs')
    return validate(json.loads((stage/'request.json').read_text())),inventory
