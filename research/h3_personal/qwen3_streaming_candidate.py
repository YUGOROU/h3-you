"""Isolated official vLLM-Omni Qwen3 candidate; NOT GPU-qualified.

Interface::
    tts = Qwen3StreamingCandidate(reference_wav, reference_text,
                                  expected_reference_sha256=voice_sha)
    for pcm_s16le_mono in tts.stream(full_japanese_text): ...
    # tts.sample_rate == 24000; chunks are bytes, NOT float arrays.

No model load, server launch, download, or third-party inference occurs here.
The parent must start ONE local server, authenticate/isolate it, stage the same
private reference, verify installed source/dependencies and enforce job expiry.
A local server is trusted with the reference; do not tunnel/forward this port.

Public source audit (immutable URLs; Apache-2.0 model and both code projects):
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc
https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py
  generate_voice_clone generates all codes, then speech_tokenizer.decode:
  NOT a native waveform iterator. No completed-WAV slicing fallback here.
https://github.com/vllm-project/vllm-omni/blob/7d2240492022c42f59158f6596f944d07243622d/examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py
https://github.com/vllm-project/vllm-omni/blob/7d2240492022c42f59158f6596f944d07243622d/vllm_omni/entrypoints/openai/serving_speech_stream.py
  stream_audio invokes _prepare_speech_generation and streams decoded PCM;
  audio.start declares native sample_rate/channels. split_granularity=none
  gives one synthesis request per input.done, not sentence synthesis.
https://github.com/vllm-project/vllm-omni/blob/7d2240492022c42f59158f6596f944d07243622d/vllm_omni/deploy/qwen3_tts.yaml
  Official async_chunk/shared-memory talker->code2wav, codec_streaming=true,
  first chunk 1 frame / steady 25 / left context 72; both stages devices=0.
  Verified upstream on H100, NOT L4. CUDA MRV2 is experimental.

Server recipe (parent responsibility, no automatic execution): check out the
OMNI_REVISION below; use that checkout's vllm_omni/deploy/qwen3_tts.yaml:
  vllm-omni serve /ABS/PINNED_MODEL_SNAPSHOT --served-model-name
  Qwen/Qwen3-TTS-12Hz-0.6B-Base --deploy-config <above YAML>
  --host 127.0.0.1 --port 8091 --omni
Download the ENTIRE model snapshot at MODEL_REVISION (includes embedded
speech_tokenizer weights), then run HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1.
No mutable repo-name load, external reference URL, quantization or 1.7B fallback.
If standalone tokenizer is needed, pin TOKENIZER_REVISION; never fetch main.

Requirements / blockers: client Python>=3.11, websockets==16.0 (lazy import).
Server source is pinned, but NOT a resolved CUDA dependency lock: upstream
requirements/common.txt uses transformers>=5.16.1,<5.20, diffusers==0.40.0,
accelerate==1.12.0, kernels==0.16.1; setup.py selects platform dependencies.
Do NOT install this broad server stack into the parent environment. A matching
Linux CUDA/vLLM/PyTorch resolved image/lock is a runtime gate, not claimed here.
Exact 0.6B + ICL reference + Japanese + L4 memory/attention backend/quality and
first waveform BEFORE generation EOS require actual parent GPU verification.
Protocol mocks establish no such acceptance. Binary frames alone cannot prove
incremental compute: retain server chunk/EOS timing evidence in live validation.
The client never retries, replays, pads, resamples, or saves/logs private audio.
"""
# GPU experiment explicitly selects official release0.30.0/sourcea8576ccb
# (runtime module verifies its shipped hashes), not unreleased OMNI_REVISION.
# Release stream handler re-audited: same protocol, audio.start omits channels;
# canonical raw PCM is mono24kHz. Explicit x-vector-only is supported when no
# transcript exists; never silently switch from requested ICL on failure.
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from pathlib import Path
import time
from urllib.parse import urlsplit
import wave

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
TOKENIZER_REVISION = "7dd38ad4e9bad454aae9cd937d0cd577604fe229"
OMNI_REVISION = "7d2240492022c42f59158f6596f944d07243622d"
QWEN_SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"


class Qwen3StreamingCandidate:
    """Synchronous, pull-based PCM16 iterator over a trusted loopback server."""

    sample_rate = 24000

    def __init__(self, reference_wav, reference_text=None, *, expected_reference_sha256,
                 x_vector_only=False, reference_transport='data-url',
                 url="ws://127.0.0.1:8091/v1/audio/speech/stream",
                 max_new_tokens=512, timeout=120.0, _connect=None):
        parsed = urlsplit(url)
        if (parsed.scheme != "ws" or parsed.hostname != "127.0.0.1"
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1/audio/speech/stream" or not parsed.port):
            raise ValueError("Only explicit IPv4 loopback speech WebSocket allowed")
        if type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be 2..1024")
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("timeout must be finite, 0..300 seconds")
        if type(x_vector_only) is not bool:
            raise ValueError('explicit cloning mode required')
        if x_vector_only:
            if reference_text is not None: raise ValueError('x-vector-only requires no transcript')
        elif not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 4096:
            raise ValueError("Reference transcript required (no implicit x-vector-only change)")
        path = Path(reference_wav)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Reference exceeds 16 MiB")
        audio = path.read_bytes()
        if hashlib.sha256(audio).hexdigest() != expected_reference_sha256:
            raise ValueError("Private reference hash mismatch")
        # Validate container, but keep original bytes/rate/channel content untouched.
        import io
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getnframes() <= 0 or wav.getcomptype() != "NONE":
                raise ValueError("Nonempty uncompressed WAV required")
        if reference_transport not in ('data-url','local-file'):
            raise ValueError('explicit reference transport required')
        reference_uri=(path.resolve(strict=True).as_uri() if reference_transport=='local-file'
                       else 'data:audio/wav;base64,'+base64.b64encode(audio).decode('ascii'))
        self._config = dict(type="session.config", model=MODEL_ID, task_type="Base",
                            language="Japanese", response_format="pcm", stream_audio=True,
                            split_granularity="none", word_timestamps=False,
                            ref_audio=reference_uri,
                            ref_text=reference_text, x_vector_only_mode=x_vector_only, seed=42,
                            max_new_tokens=max_new_tokens)
        if len(json.dumps(self._config,ensure_ascii=False).encode())>4*1024*1024:
            raise ValueError('official session.config 4MiB bound: select local-file explicitly')
        self._url, self._timeout, self._connect = url, timeout, _connect
        self._busy = False

    def stream(self, text):
        """Yield native mono s16le bytes immediately; close iterator on cancellation."""
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError("Full nonempty text required, at most 4096 characters")
        if self._busy:
            raise RuntimeError("Only one active utterance allowed")
        self._busy = True
        try:
            connect = self._connect
            if connect is None:
                from websockets.sync.client import connect
            deadline = time.monotonic() + self._timeout
            with connect(self._url, open_timeout=self._timeout, close_timeout=2,
                         max_size=2 * 1024 * 1024, max_queue=1, proxy=None) as ws:
                for message in (self._config, {"type": "input.text", "text": text},
                                {"type": "input.done"}):
                    ws.send(json.dumps(message, ensure_ascii=False))
                started = done = False
                total = 0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("TTS utterance deadline exceeded")
                    frame = ws.recv(timeout=remaining)
                    if isinstance(frame, bytes):
                        if not started or done or not frame or len(frame) % 2:
                            raise RuntimeError("Invalid PCM frame/order")
                        total += len(frame)
                        if total > self.sample_rate * 2 * 120:
                            raise RuntimeError("PCM exceeds finite output bound")
                        yield frame
                        continue
                    msg = json.loads(frame)
                    kind = msg.get("type")
                    if kind == "error":
                        message=str(msg.get('message',''))
                        # Bounded private diagnostic, never raw reference/base64.
                        message=message.replace(self._config['ref_audio'],'<private-reference>')
                        message=re.sub(r'[A-Za-z0-9+/=_-]{80,}','<encoded-data-redacted>',message)
                        self.last_failure={'message':message[:4096],'received_pcm_bytes':total,
                                           'audio_started':started}
                        raise RuntimeError("Local TTS generation failed (details withheld)")
                    if msg.get("utterance_index") != 0:
                        raise RuntimeError("Unexpected utterance index")
                    if kind == "audio.start":
                        if (started or msg.get("sentence_index") != 0 or msg.get("format") != "pcm"
                                or msg.get("sample_rate") != self.sample_rate or msg.get("channels", 1) != 1):
                            raise RuntimeError("Unexpected native audio format/order")
                        started = True
                    elif kind == "audio.done":
                        if (not started or done or not total or msg.get("sentence_index") != 0
                                or msg.get("error") or msg.get("total_bytes") != total):
                            raise RuntimeError("Incomplete/failed native audio")
                        done = True
                    elif kind == "session.done":
                        if not done or msg.get("total_sentences") != 1:
                            raise RuntimeError("Expected exactly one completed synthesis")
                        return
                    else:
                        raise RuntimeError("Unexpected streaming protocol frame")
        finally:
            self._busy = False
