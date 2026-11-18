"""One native FP32 Irodori L4 owner. No H3, source changes, or device playback."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def main():
    os.umask(0o077)
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    expires = int(os.environ['H3_TTS_EXPIRES_AT'])
    if not 10 < expires-time.time() <= 1200:
        raise RuntimeError('expired or unbounded endpoint')
    if len(os.environ.get('H3_TTS_KEY', '')) < 32 or not os.environ.get('HF_TOKEN'):
        raise RuntimeError('owned secrets required')
    key, session = os.environ['H3_TTS_KEY'], os.environ['H3_TTS_SESSION']
    os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1',
                      H3_TTS_MEASURE_MEMORY='1', UV_NO_CACHE='1')

    def expired(*_): raise TimeoutError('finite bootstrap expired')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(expires-time.time()-5)))
    tts = None
    try:
        # Pinned bootstrap-only dependency; native dependencies use original uv.lock.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir',
                        'huggingface_hub==1.7.2'], check=True,
                       timeout=max(1, expires-time.time()-10))
        if any(shutil.which(name) is None for name in ('git', 'ffmpeg', 'ffprobe')):
            subprocess.run(['apt-get', 'update', '-qq'], check=True,
                           timeout=max(1, expires-time.time()-10))
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'git', 'ffmpeg', 'libsndfile1'],
                           check=True, timeout=max(1, expires-time.time()-10))
        if any(shutil.which(name) is None for name in ('git', 'ffmpeg', 'ffprobe')):
            raise RuntimeError('native runtime executables unavailable')
        from research.hf_h3_live_voice import prepare_tts
        from research.h3_personal.director_tts_job import SOURCE, MODEL, CODEC, REFERENCE_SHA
        from research.h3_personal.irodori_endpoint_service import serve
        workspace = Path('/tmp/h3-irodori-endpoint')
        workspace.mkdir(mode=0o700, exist_ok=False)
        tts = prepare_tts(root, {'live_direct_session': {}}, workspace=workspace)
        native = tts.read('tts_ready', time.monotonic()+max(.001, expires-time.time()-5))
        metrics = native['metrics']
        if (metrics['device_name'] != 'NVIDIA L4' or metrics['device_count'] != 1
                or metrics['model_all_cuda_fp32'] is not True
                or metrics['codec_all_cuda_fp32'] is not True):
            raise RuntimeError('actual one-L4/native FP32 required; no fallback')
        ready = dict(ready=True, device_name=metrics['device_name'], device_count=1,
                     model_precision='fp32', codec_precision='fp32', compile_model=False,
                     watermark_ready=True, source_revision=SOURCE, model_revision=MODEL,
                     codec_revision=CODEC, reference_sha256=REFERENCE_SHA,
                     verified_cuda_fp32_parameters=True, memory=metrics)
        os.environ.pop('HF_TOKEN', None)
        signal.alarm(0)
        print(json.dumps(dict(type='h3_irodori_endpoint_ready', device_name='NVIDIA L4',
                              device_count=1, memory=metrics)), flush=True)
        serve(tts, ready, key=key, session=session, expires_at=expires, max_requests=2)
    finally:
        signal.alarm(0)
        if tts is not None: tts.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Sanitized diagnostics only; upstream subprocess stderr never contains argv secrets.
        print(json.dumps(dict(type='h3_irodori_endpoint_failed', error_type=type(exc).__name__)), flush=True)
        raise SystemExit(1)
