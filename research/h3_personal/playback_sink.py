"""Opt-in Mac/local buffered A/V sink using one ffplay process per clip.

No GPU/server/OBS is started. Video and original PCM TTS are remuxed without
re-encoding/resampling into one private Matroska file. Stop kills/reaps its process
group; physical output already emitted to OS/devices cannot be recalled.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from .buffered_stream import _asset
from .e2e import probe


class PlaybackInterrupted(RuntimeError):
    pass


def _kill(process):
    if process is None:
        return
    if process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=2)
    else:
        process.wait()


def _snapshot(root, name, expected, destination):
    source = _asset(root, name, expected)
    digest = hashlib.sha256()
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as src, destination.open('xb') as dst:
        destination.chmod(0o600)
        while block := src.read(1024*1024):
            digest.update(block); dst.write(block)
    if digest.hexdigest() != expected:
        raise ValueError('playback snapshot bytes differ')


class LocalPlaybackSink:
    def __init__(self, root, *, allow_playback=False):
        if allow_playback is not True:
            raise ValueError('local audiovisual playback requires explicit opt-in')
        if not shutil.which('ffmpeg') or not shutil.which('ffplay'):
            raise ValueError('ffmpeg/ffplay required; no automatic installation')
        self.root = Path(root).resolve(strict=True)
        self._lock = threading.RLock(); self._generation = 0
        self._working = False; self._process = None; self._temp = None
        self._closed = False

    def _check(self, generation, is_current):
        if self._closed or generation != self._generation or not is_current():
            raise PlaybackInterrupted('playback epoch invalidated')

    def _spawn(self, command, generation, is_current):
        with self._lock:
            self._check(generation, is_current)
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return self._process

    def _cleanup(self):
        if self._temp is not None:
            self._temp.cleanup(); self._temp = None

    def start(self, item, *, is_current):
        with self._lock:
            if self._closed or self._working or self._process is not None:
                raise RuntimeError('sink closed or busy; poll/stop before next clip')
            generation = self._generation
            self._check(generation, is_current)
            self._working = True
            self._temp = tempfile.TemporaryDirectory(prefix='h3-playback-')
            directory = Path(self._temp.name)
        try:
            video = directory/'video'; audio = directory/'tts'; mux = directory/'playback.mkv'
            _snapshot(self.root,item.clip.video,item.clip.video_sha256,video)
            _snapshot(self.root,item.clip.original_tts,item.clip.audio_sha256,audio)
            with self._lock: self._check(generation,is_current)
            # PCM copy preserves the original TTS samples; reject other formats.
            streams = [s for s in probe(audio)['streams'] if s['codec_type']=='audio']
            if len(streams)!=1 or streams[0]['codec_name'] not in ('pcm_s16le','pcm_s24le','pcm_s32le'):
                raise ValueError('original PCM TTS required; no implicit audio conversion')
            from fractions import Fraction
            if Fraction(streams[0]['duration']) > item.duration:
                raise ValueError('refusing TTS truncation')
            command = ['ffmpeg','-v','error','-nostdin','-i',str(video),'-i',str(audio),
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','copy',str(mux)]
            process = self._spawn(command,generation,is_current)
            if process.wait()!=0:
                with self._lock: self._check(generation,is_current)
                raise RuntimeError('private A/V remux failed')
            info=probe(mux)
            if abs(Fraction(info['format']['duration']) - item.duration) > Fraction(1,24):
                raise ValueError('remux duration differs from validated playback')
            kinds=[s['codec_type'] for s in info['streams']]
            if sorted(kinds)!=['audio','video']:
                raise ValueError('remux must contain one video and original TTS only')
            for s in info['streams']:
                if abs(Fraction(s.get('start_time','0'))) > Fraction(1,24):
                    raise ValueError('remux has nonzero A/V start')
            self._spawn(['ffplay','-v','error','-nostats','-autoexit','-window_title',
                'AI-generated H3 avatar — buffered',str(mux)],generation,is_current)
        except BaseException:
            with self._lock:
                _kill(self._process); self._process=None; self._cleanup()
            raise
        finally:
            with self._lock: self._working=False

    def poll(self):
        """None while not finished; process returncode once finished and reaped."""
        with self._lock:
            if self._working or self._process is None:
                return None
            code=self._process.poll()
            if code is not None:
                self._process.wait(); self._process=None; self._cleanup()
            return code

    def stop(self):
        with self._lock:
            self._generation+=1
            _kill(self._process); self._process=None
            # During snapshot/remux, the start owner cleans up after noticing cancel.
            if not self._working: self._cleanup()

    def close(self):
        with self._lock:
            self._closed=True
        self.stop()

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


class PlaybackPump:
    """One tick producer; interrupt may be called from another control thread."""
    def __init__(self, queue, sink):
        self.queue=queue; self.sink=sink; self._item=None
        self._lock=threading.Lock(); self._ticking=False

    def tick(self):
        with self._lock:
            if self._ticking: raise RuntimeError('overlapping playback ticks refused')
            self._ticking=True
        try:
            item=self._item
            if item is None:
                item=self.queue.take()
                if item is None: return False
                self._item=item
                try: self.sink.start(item,is_current=lambda:self.queue.current(item.turn))
                except PlaybackInterrupted:
                    self.interrupt(); return False
                except BaseException:
                    self.interrupt(); raise
            if not self.queue.current(item.turn):
                self.sink.stop(); self._item=None; return False
            code=self.sink.poll()
            if code is None: return False
            self._item=None
            if code!=0:
                self.interrupt(); raise RuntimeError('local playback failed')
            return self.queue.acknowledge(item)
        finally:
            with self._lock: self._ticking=False

    def interrupt(self):
        # Invalidate producer first, then stop/flush both tracks in the one process.
        self.queue.interrupt(); self.sink.stop(); self._item=None

    def close(self):
        self.queue.interrupt(); self.sink.close(); self._item=None
