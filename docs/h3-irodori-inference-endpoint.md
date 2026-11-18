# Native Irodori standalone Inference Endpoint

Current hardware: USER-approved AWS us-east-1 L4×1, authenticated Endpoint/max1replica. H3 placement/license gates remain separate. Native FP32/eager/40steps/original pins/reference/SilentCipher unchanged. Single model owner, sequential calls; no quantization/offload/local fallback. Standalone result: `h3-irodori-l4-endpoint-validation-20261028-v2.md`.

## Usage
From repo root (existing HF uv-tool Python, no root dependency changes):
```
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_irodori_endpoint
# Explicit cloud operation, fresh private directory only:
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_irodori_endpoint --run --directory FRESH_PRIVATE_DIRECTORY
```
Offline default does not create/query/resume a resource. `--run` creates a fresh private immutable code repository and one unique Endpoint, tests two original native utterances, and pauses in finally. It does not launch H3 or devices. Native dependencies use the original frozen uv.lock/cu128; bootstrap reuses accepted native preparation. Repository contains code/notices/manifest, not voice input/outputs or credentials.

A local namespace-wide owner lock prevents parallel controller invocations; Endpoint inventory also rejects another unpaused owned TTS service. Persistent unresolved-create fence remains after uncertain cleanup, blocking another launch. A404 after an uncertain POST is not proof of non-creation. Creation/synthesis never automatically replay. Pausing is idempotent and checked through bounded control retries.

App expiry1200s/startup wait≤900s/per-request≤120s. Native child/service stops at expiry. **Container exit does not prove Endpoint allocation/billing stopped.** External owner must confirm provider paused; failed control/API access is explicit cleanup-unconfirmed/operator action, not a guaranteed hard cloud-stop. No detached watcher or automatic lease extension.

## HTTP boundary
Platform authentication plus secret X-H3-TTS-Key. Secrets only in protected environment/in-memory headers, never argv/config files/logs. GET /health exposes minimal readiness for platform probes; GET /ready requires app key and returns native source/model/codec/reference/device/precision proof. Busy synthesis does not fail health.

POST /synthesize exact JSON: session, strictly increasing epoch, text≤160characters. One nonblocking native slot, at most2 requests for this initial validation service, bounded4 handlers. Request headers/body deadlines and sizes are bounded; duplicate JSON fields/replayed epoch rejected. Real synthesis errors/unknown result quarantine/close the child, no retry or silent fallback.

Response contains complete original PCM16 WAV as bounded base64 plus native receipt/allowlisted metrics. This is **complete-WAV TTS, not PCM streaming**. Service verifies WAV/hash/pins/watermark and deletes its generated tmp directory after reading. Client independently verifies native proof/format/sample/hash/text/epoch and creates private exclusive output. Readiness checks actual single L4 and FP32/CUDA model+codec parameters; metrics are allocator values, not all-device memory accounting.

RemoteIrodoriTTS follows the existing `read('tts_ready', deadline)`, `synthesize(text,turn_id,directory,deadline)`, `close()` seam. It rejects non-HF HTTPS hosts/redirects, caps responses, updates absolute socket deadlines and avoids redundant EOF/deadline reads after framed completion. close interrupts current I/O. Its close does **not** pause the provider: the external resource owner must do that.

## H3 integration is a separate gate
The current H3 deployment still loads local TTS by default. A selected Endpoint mode must be immutable/advertised, obtain the app key through secrets, verify remote readiness/boot/session/expiry and **skip local native TTS preparation/model/child entirely**. Tests must prove no fallback to co-location on failure. Couple resource cleanup without unnecessarily restarting healthy H3 for a recoverable front-end/TTS failure. Do not resume the validated paused Endpoint with an expired lease or unknown discarded app key.

## Tests
```
~/.local/share/uv/tools/hf/bin/python -m unittest discover -s tests -p 'test_h3_irodori_endpoint*.py' -q
```
Additional native/install regressions: test_h3_director_irodori.py, test_h3_conversation_tts_install.py. CPU fixtures prove boundaries only; device/memory observations come from actual L4 acceptance. Native mark application is not newly verified watermark detection/robustness or perceptual voice acceptance.
