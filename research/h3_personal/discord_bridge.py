"""Official-bot transport core, no Discord/model/network imports at import time.

One owning model thread; bounded one turn, epoch cancellation, no auto retry.
This is TEXT-in / buffered media-out, not ASR or bot Go Live video streaming.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from .buffered_stream import Clip,validate_clip
from .component_cache import sha256
from .generated_transfer import H3_REVISION


@dataclass(frozen=True)
class DiscordScope:
    guild_id:int
    channel_id:int
    user_id:int

    def __post_init__(self):
        if any(type(x) is not int or x<=0 for x in (self.guild_id,self.channel_id,self.user_id)):
            raise ValueError('explicit positive Discord guild/channel/user IDs required')

    def allows(self,guild,channel,user,*,is_bot=False):
        return not is_bot and (guild,channel,user)==(self.guild_id,self.channel_id,self.user_id)


def mux_generated_bundle(manifest,adapter_sha256,output):
    """Verified GENERATED-only private bundle -> bounded AAC review attachment.

    Original PCM remains in bundle; Discord MP4 is explicitly a lossy review copy.
    """
    manifest=Path(manifest);root=manifest.parent
    if manifest.is_symlink() or manifest.stat().st_size>32768:raise ValueError('bounded regular manifest')
    m=json.loads(manifest.read_text())
    if (m.get('schema')!='h3-generated-playback-transfer-v1' or m.get('ai_generated') is not True or
        m.get('h3_revision')!=H3_REVISION or m.get('adapter_sha256')!=adapter_sha256 or
        m.get('audio_source')!='voxcpm2_zero_shot_generated' or m.get('video_source')!='h3_ref2va_identity'):
        raise ValueError('generated personalized H3/TTS only')
    assets=m['assets'];v=assets['video'];a=assets['original_tts']
    if v['file']!='video.mp4' or a['file']!='tts.wav':raise ValueError('fixed assets only')
    for meta in (v,a):
        p=root/meta['file']
        if p.is_symlink() or p.stat().st_size!=meta['bytes'] or not 0<meta['bytes']<=64*1024**2 or sha256(p)!=meta['sha256']:
            raise ValueError('generated bundle bytes differ')
    duration=validate_clip(root,Clip('video.mp4','tts.wav',v['sha256'],a['sha256']))
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('no review overwrite')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(root/'video.mp4'),'-i',str(root/'tts.wav'),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-af','apad',
        '-t',str(float(duration)),'-movflags','+faststart',str(output)],check=True)
    if not 0<output.stat().st_size<=8*1024**2:raise ValueError('bounded8MiB Discord attachment required')
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'],check=True)
    output.chmod(0o600)
    return output


class DiscordTurnBridge:
    """Factory runs ON the persistent owning thread, never asyncio.to_thread pool.

    deliver(review_path) is asynchronous and returns a Discord message-like handle.
    cancel invalidates work cooperatively; already emitted client media can't be recalled.
    """
    def __init__(self,scope,producer_factory,root,adapter_sha256,*,mux=mux_generated_bundle):
        self.scope=scope;self.factory=producer_factory;self.root=Path(root);self.adapter=adapter_sha256
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_mode&0o077:raise ValueError('private bridge root required')
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='h3-owner')
        self._producer=None;self._task=None;self._epoch=0;self._cancel=threading.Event()
        self._closed=False;self._last_message=None;self.mux=mux

    def submit(self,guild,channel,user,text,deliver,*,is_bot=False):
        if self._closed:raise RuntimeError('bridge closed')
        if not self.scope.allows(guild,channel,user,is_bot=is_bot):return 'forbidden'
        if not isinstance(text,str) or not 1<=len(text.strip())<=4096:return 'invalid'
        if self._task is not None and not self._task.done():return 'busy'
        self._epoch+=1;epoch=self._epoch;event=threading.Event();self._cancel=event
        self._task=asyncio.create_task(self._run(text,event,epoch,deliver))
        return 'accepted'

    def _render(self,text,event,epoch):
        if self._producer is None:self._producer=self.factory()
        before=list(self._producer.history)
        self._history_before=before
        target=self.root/('turn-'+str(epoch))
        try:
            manifest=self._producer.produce(text,target,is_current=lambda:not event.is_set())
            if event.is_set():raise RuntimeError('stale turn')
            return self.mux(manifest,self.adapter,target/'discord-review.mp4')
        except BaseException:
            self._producer.history[:]=before
            raise

    def _rollback(self):
        if self._producer is not None:self._producer.history[:]=self._history_before

    async def _run(self,text,event,epoch,deliver):
        loop=asyncio.get_running_loop()
        try:
            review=await loop.run_in_executor(self._executor,self._render,text,event,epoch)
            if event.is_set():
                await loop.run_in_executor(self._executor,self._rollback)
                return 'interrupted'
            message=await deliver(review)
            if event.is_set():
                # HTTP send may have finished after cancellation. Best-effort remove OWN post.
                try:await message.delete()
                finally:await loop.run_in_executor(self._executor,self._rollback)
                return 'interrupted'
            self._last_message=message
            return 'delivered'
        except Exception:
            await loop.run_in_executor(self._executor,self._rollback)
            return 'interrupted' if event.is_set() else 'failed'  # Never echo exception/text/token.

    async def wait_result(self):
        return await asyncio.shield(self._task) if self._task is not None else 'idle'

    async def stop(self,guild,channel,user):
        if not self.scope.allows(guild,channel,user):return 'forbidden'
        self._cancel.set()
        if self._last_message is not None:
            message=self._last_message;self._last_message=None
            try:await message.delete()
            except Exception:pass  # Not physical client flush.
        return 'interrupted'

    async def close(self):
        if self._closed:return
        self._closed=True;self._cancel.set()
        if self._task is not None:await self._task
        loop=asyncio.get_running_loop()
        def dispose():
            if self._producer is not None:
                session=getattr(self._producer.h3,'session',None)
                if session is not None:session.close()
        await loop.run_in_executor(self._executor,dispose)
        self._executor.shutdown(wait=True)
