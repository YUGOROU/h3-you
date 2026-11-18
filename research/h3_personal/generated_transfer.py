"""Pinned private generated-only playback handoff. No SDK/GPU at import.

Producer labels are assertions, not independent proof of generation or lip sync.
Only manifest+two fixed assets in generated-playback namespace are downloadable.
"""
import hashlib
import json
from pathlib import Path
import os
import re
import shutil
import tempfile
from .component_cache import canonical, sha256, digest_ok
from .buffered_stream import Clip, validate_clip

H3_REVISION='42ed227ee7df40d41602854ae760620d6eb651fe'


class TransferInterrupted(RuntimeError):
    pass


def fetch(*, api, download, repo, revision, bundle, manifest_sha256,
          adapter_sha256, destination, is_current=lambda:True, validator=validate_clip):
    """download(repo,path,revision,target) must write only target; single finite batch."""
    if (not re.fullmatch(r'YUGOROU/[A-Za-z0-9_.-]+',repo) or
        not re.fullmatch(r'[0-9a-f]{40}',revision) or
        not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',bundle) or
        not digest_ok(manifest_sha256) or not digest_ok(adapter_sha256)):
        raise ValueError('private owner, immutable pins and safe bundle required')
    def check():
        if not is_current():raise TransferInterrupted('stale playback transfer discarded')
    check()
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise ValueError('never overwrite a transfer')
    parent=destination.parent;parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if parent.is_symlink() or parent.stat().st_mode & 0o077:
        raise ValueError('private destination parent required')
    info=api.repo_info(repo,repo_type='model',revision=revision,files_metadata=True)
    if info.private is not True or info.sha!=revision:raise ValueError('private pinned repo required')
    remote={s.rfilename:s for s in info.siblings}
    prefix='generated-playback/'+bundle+'/'
    manifest_path=prefix+'manifest.json'
    if manifest_path not in remote or not 0<remote[manifest_path].size<=32768:
        raise ValueError('small generated-playback manifest required')
    staging=Path(tempfile.mkdtemp(prefix='.transfer-',dir=parent))
    try:
        def pull(name,target,expected,size):
            check();download(repo,name,revision,target);check()
            if target.is_symlink() or not target.is_file() or target.stat().st_size!=size or sha256(target)!=expected:
                raise ValueError('transfer bytes/size differ')
            target.chmod(0o600)
        pull(manifest_path,staging/'TRANSFER.json',manifest_sha256,remote[manifest_path].size)
        manifest=json.loads((staging/'TRANSFER.json').read_text())
        if (set(manifest)!={'schema','ai_generated','audio_source','video_source','h3_revision','adapter_sha256','assets'} or
            manifest['schema']!='h3-generated-playback-transfer-v1' or manifest['ai_generated'] is not True or
            manifest['audio_source']!='voxcpm2_zero_shot_generated' or manifest['video_source']!='h3_ref2va_identity' or
            manifest['h3_revision']!=H3_REVISION or manifest['adapter_sha256']!=adapter_sha256 or
            set(manifest['assets'])!={'video','original_tts'}):
            raise ValueError('generated H3/identity/TTS provenance contract required; raw media refused')
        assets=manifest['assets'];total=0
        for kind,name in (('video','video.mp4'),('original_tts','tts.wav')):
            a=assets[kind]
            if (set(a)!={'file','sha256','bytes'} or a['file']!=name or not digest_ok(a['sha256']) or
                type(a['bytes']) is not int or not 0<a['bytes']<=64*1024*1024):
                raise ValueError('fixed generated asset names and bounded sizes required')
            s=remote.get(prefix+name)
            if s is None or s.size!=a['bytes'] or (s.lfs and s.lfs.sha256!=a['sha256']):
                raise ValueError('pinned remote asset metadata differs')
            total+=a['bytes']
        if total>80*1024*1024:raise ValueError('transfer batch exceeds80MiB')
        # Verify BOTH descriptors before retrieving either asset.
        for kind,name in (('video','video.mp4'),('original_tts','tts.wav')):
            a=assets[kind];pull(prefix+name,staging/name,a['sha256'],a['bytes'])
        clip=Clip('video.mp4','tts.wav',assets['video']['sha256'],assets['original_tts']['sha256'])
        duration=validator(staging,clip)
        if not 0<duration<=15:raise ValueError('bounded playback duration required')
        (staging/'clips.json').write_bytes(canonical({'schema':'h3-buffered-playback-v1','clips':[clip.__dict__]}))
        (staging/'RECEIPT.json').write_bytes(canonical({'repo':repo,'revision':revision,'bundle':bundle,
            'manifest_sha256':manifest_sha256,'adapter_sha256':adapter_sha256,
            'producer_claims_independently_verified':False,'media_bytes_verified':True,'autoplay':False}))
        for p in staging.iterdir():p.chmod(0o600)
        check()
        # Destination never reused. Nonempty directory publication is atomic.
        if destination.exists() or destination.is_symlink():raise ValueError('destination appeared during transfer')
        os.rename(staging,destination)
        return destination/'clips.json'
    finally:
        if staging.exists():shutil.rmtree(staging)


def fetch_into_queue(queue, turn, **kwargs):
    """Bind transfer to the existing assistant epoch; no playback/device start."""
    if not queue.current(turn):return False
    destination=queue.root/kwargs['bundle']
    fetch(destination=destination,is_current=lambda:queue.current(turn),**kwargs)
    local=json.loads((destination/'clips.json').read_text())['clips'][0]
    local['video']=kwargs['bundle']+'/'+local['video']
    local['original_tts']=kwargs['bundle']+'/'+local['original_tts']
    return queue.enqueue(turn,Clip(**local))
