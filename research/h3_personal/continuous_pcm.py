"""One bounded raw integer PCM -> canonical PCM FFmpeg lifetime per utterance.

The caller owns receipt validation. No quantization precedes FFmpeg. The reader
is the sole output callback owner; input writes never wait on consumer credits.
"""
import math
import os
import select
import subprocess
import tempfile
import threading
import time
import wave

MAX_BYTES = 2880000


def native_wav(path):
    """Call only with a receipt-validated path; preserve original integer bytes."""
    with wave.open(str(path), 'rb') as wav:
        fmt = (wav.getframerate(), wav.getsampwidth(), wav.getnchannels())
        rate, width, channels = fmt
        if not (0 < rate <= 48000 and width in (2, 3, 4)
                and channels == 1 and wav.getcomptype() == 'NONE'):
            raise ValueError('bounded mono integer PCM WAV required')
        frames = wav.getnframes()
        if not 0 < frames <= rate * 30:
            raise ValueError('source duration exceeds 30s or empty')
        raw = wav.readframes(frames)
        if len(raw) != frames * width:
            raise ValueError('truncated source PCM')
        return fmt, raw


class ContinuousPCM:
    def __init__(self, fmt, root, deadline, check, emit):
        rate, width, channels = fmt
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError('finite future deadline required')
        if not (all(type(v) is int for v in fmt)
                and 0 < rate <= 48000 and width in (2, 3, 4) and channels == 1):
            raise ValueError('unsupported native format')
        self.fmt, self.deadline, self.check, self.emit = fmt, deadline, check, emit
        self.error = None
        self.stop = threading.Event()
        self.source_bytes = self.output_bytes = 0
        self.private = tempfile.TemporaryDirectory(dir=root)
        self.process = self.reader = None
        try:
            self.process = subprocess.Popen([
                'ffmpeg', '-nostdin', '-v', 'error', '-xerror',
                '-f', f's{width*8}le', '-ar', str(rate), '-ac', '1',
                '-probesize', '32', '-analyzeduration', '0', '-i', 'pipe:0',
                '-map', '0:a:0', '-ac', '1', '-ar', '48000',
                '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'],
                cwd=self.private.name, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.reader = threading.Thread(target=self._read, name='h3-continuous-pcm')
            self.reader.start()
        except BaseException:
            self.stop.set()
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=2)
                self.process.stdin.close()
                self.process.stdout.close()
            if self.reader is not None and self.reader.ident is not None:
                self.reader.join(timeout=2)
            self.private.cleanup()
            raise

    def _check(self):
        if self.error is not None:
            raise self.error
        if self.stop.is_set():
            raise InterruptedError('resampler stopped')
        self.check()
        if time.monotonic() >= self.deadline:
            raise TimeoutError('resampler deadline')

    def _read(self):
        pending = b''
        try:
            fd = self.process.stdout.fileno()
            while True:
                self._check()
                if not select.select([fd], [], [], .02)[0]:
                    continue
                body = os.read(fd, 64000)
                if not body:
                    if pending:
                        raise ValueError('partial canonical sample at EOS')
                    return
                body = pending + body
                end = len(body) & ~1
                pending = body[end:]
                if end:
                    self.output_bytes += end
                    if self.output_bytes > MAX_BYTES:
                        raise BufferError('canonical PCM exceeds 30s')
                    self.emit(body[:end])
        except BaseException as exc:
            self.error = exc

    def feed(self, fmt, raw):
        self._check()
        if fmt != self.fmt:
            raise ValueError('native sentence format drift')
        self.source_bytes += len(raw)
        if self.source_bytes > self.fmt[0] * self.fmt[1] * 30:
            raise BufferError('aggregate source exceeds 30s')
        view = memoryview(raw)
        fd = self.process.stdin.fileno()
        while view:
            self._check()
            if select.select([], [fd], [], .02)[1]:
                try:
                    count = os.write(fd, view[:64000])
                except BlockingIOError:
                    continue
                view = view[count:]

    def finish(self):
        self._check()
        self.process.stdin.close()  # the only real EOS
        while self.reader.is_alive() or self.process.poll() is None:
            self._check()
            self.reader.join(.02)
        self._check()
        if self.process.returncode or not self.output_bytes:
            raise ValueError('FFmpeg failed or empty canonical PCM')

    def close(self):
        self.stop.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=2)
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.private.cleanup()
        if self.reader.is_alive():
            raise RuntimeError('resampler reader did not terminate')
