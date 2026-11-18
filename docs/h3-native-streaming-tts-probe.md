# Isolated native streaming TTS L4 probe

USER authorizes VoxCPM2 and Qwen3-TTS-12Hz-0.6B-Base implementation/CPU tests/actual finite GPU Job evaluation. This is NOT adoption, H3 migration, playback or device permission. Original Irodori/H3 defaults remain unchanged.

## Entry point
```
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_native_tts_probe
# Explicit actual compute, ONE candidate, fresh private external directory:
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_native_tts_probe --run --candidate voxcpm2 --directory FRESH_DIRECTORY
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_native_tts_probe --run --candidate qwen3 --directory ANOTHER_FRESH_DIRECTORY
```
Run sequentially only after previous provider termination. CLI offline default has no network/cloud action. One dedicated actual NVIDIA L4, hardware/rate/license/private-reference gates, timeout1200s/attempts1. Local exclusive lock + persistent unknown-create fence; one creation POST, no automatic replay. Model/source/image/artifact hashes verified. Controller cancels/reconciles in finally; unresolved resource blocks another launch.

### Explicit Vox compile and telemetry control

Add `--voxcpm-optimize` only for a separate upstream compile experiment (eager remains default). Requires all four compile wrappers and no suppress-errors fallback. Compile-only bootstrap provisions gcc/libc6-dev in the disposable Job if absent, bounded by install timeouts; records compiler version. These OS packages are repository-resolved, not hash-pinned; Python/source/model pins stay unchanged. No local environment installation.

Add `--voxcpm-trace-abba` together with that flag for two bounded OFF/ON warmups then OFF/ON/ON/OFF same-text measurements on one resident model. Seed42 reset does not guarantee identical PCM; compare hashes. Warmup audio is deleted, summaries retained. Default four-text probe remains unchanged.

Vox candidate `completion` observes the pinned native stop branch with scoped Python tracing, separately from iterator completion. `formal_completion_accepted` requires natural stop and stream exhaustion, not full spoken-text/perceptual acceptance. Trace-OFF explicitly records unknown native stop. Tracing may affect throughput; use the control profile before attributing speed to compilation. See actual results and limitations.

### Targeted eager supply diagnostics

Vox-only, mutually exclusive with full compile/ABBA:
- `--voxcpm-supply-diagnostic`: eager baseline then decoder-only legacy weight_norm folding, exact fixed-latent PCM gate before measuring. Experiment only, original files untouched. Result.
- `--voxcpm-stage-diagnostic`: eager baseline plus bounded current-stream CUDA event attribution, no added synchronization. Inclusive spans include dispatch/idle/queue semantics. Result.
- Add `--voxcpm-lm-compile` to stage mode for scoped LM-forward_step-only compile and bounded warm/retained streams; feature/VAE remain eager. Compiler provisioning applies. Negative result; do not adopt.

`measure(...,timing_attribution=True)` adds host next-call/consumer spans with perf_counter, not pure GPU time. Retained artifacts and finite owner limits unchanged. No flag authorizes playback or production backend selection.

## Model/source/runtime bindings
- Vox: openbmb/VoxCPM2@32279effe8c19989596f05d353d1447f51d9e915; source cce58a5b59303c9bd63d12f23afe6a49a4a80c59, Apache2. Public source archive SHA a2b9d9f3076d9d6eaaf5fdb6b92e1bc7c3b4643bc9a489bace7622e04bcdbd2c. uv may omit PEP610 archive hash, so check all38 installed Python files against independently archive-bound tree SHA6df6c47053cb140e2400c8000786aa4127ce5a4964f7265eff2b321862768fcd. Immutable URL alone is not sufficient. Local native AudioVAE is part of model snapshot. Explicit CUDA/eager, no denoiser/normalizer/retries, single generate_streaming(fulltext, reference_wav_path). Native configuration defaults, no quantization.
- Qwen: Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc with embedded tokenizer. Official local vLLM-Omni0.30.0 image linux/amd64 sha256:b09c9eab7d2e30817208de92ccb482756d13d4339377d7cc4c4ce9f47bf04174; sourcea8576ccb725c4e21cd13c3eb5f9a546b21149d2b/shipped handler+YAML hashes checked. Do not pretend it is the newer development commit audited initially. Explicit CUDA-default MRV2/eager/BF16/single sequence; official chunk ramp[4,4,8,16,25], unchanged72-frame context, async_chunk/codec_streaming. No model/precision fallback.
- Reference remains the original private pinned Irodori reference (same SHA); new models Apache2 do not inherit H3's territory rules. User has authorized non-JP/AU TTS compute. No H3 weights/reference image involved. New candidate output is explicitly AI-generated; Irodori watermark is not claimed/preserved on these different models.

## Full-input / native-wave boundary
Vox directly yields its native generator's 1D floating waveform. Qwen native Python generate_voice_clone alone is NOT streaming; selected official vLLM-Omni pipeline streams decoded mono24k PCM over a loopback-only WebSocket. Both get ONE full text, no external sentence splitting/completed-WAV slicing.

Qwen uses explicit x-vector-only reference cloning, because the existing reference has no trusted transcript. Do not invent a transcript or describe this as ICL quality. Reference transport is file URI inside a dedicated private directory, server --allowed-local-media-path scoped only there. Its bytes/hash/rate/duration are NOT shortened/modified. Official session.config limit4MiB rejects Base64 of the actual reference before inference. Client now rejects oversized configurations on CPU rather than spending a GPU launch to discover it. Original reference WAV uses WAVE_FORMAT_EXTENSIBLE: Python3.12+ is required for stdlib wave parsing of this particular asset (current Qwen image supports it); HF CLI's Python3.11 is not the same parser.

## Isolation / output ownership
Vox dependencies locked in research/tts_streaming_experiment/voxcpm2/{pyproject.toml,uv.lock}. All new installs use isolated uv overlay/venv in GPU container; uv executable bootstrapped from hash-pinned Linux wheel without local pip/root changes (compile-only disposable OS toolchain exception above). No root/HF CLI/iOS environment install. Qwen uses the exact official image plus isolated websockets16 client overlay. Models are downloaded by immutable snapshot and each file is hash checked; native inference offline. HF token removed before inference; native server also lacks the artifact owner key.

Private deployment dataset contains only allowlisted source/lock/model manifest/public test text, not reference/generated audio/credentials. Source ZIP uses HF CDN signed redirect with bearer stripped off-origin. Generated native WAV and timing artifacts are downloaded by authenticated HF Jobs port7860 + ephemeral owner key/boot identity. Port8091 is loopback only. No generated audio publication/media upload; no device playback. Results remain outside repo.

## Explicit bounded preroll (separate from model speed)

USER prefers a finite initial reserve to interruptions. `relay_native_tts.produce(...,preroll=PCMPrerollPolicy())` implements a1second48k source threshold with observation-checked wait bounds; omitting it preserves first-PCM publication. No speed/voice/rate changes or automatic buffer growth. [Contract](h3-native-tts-preroll.md), six saved-GPU-PCM result. [Explicit native boot/worker selection](h3-native-tts-direct-backend.md) is CPU-implemented, defaultOFF; USER lifted the short GPU hold; first coupled GPU result confirms two-turn PCM/tail/native EOS but not unfinished-source H3 conditioning or gap-free playback. Do not require standalone immediate-playout RTF<1 before testing buffered H3 delivery; combine preroll with actual native waveform readiness without additive waiting.

## Measurement and limits
Four common cases: greeting/conversational/technical/repeated greeting. Model load excluded; first request and warm requests separated. Repeated text does not establish identical RNG across model stacks. No meaningful production p95 claim from three warm samples.

Collector records first nonempty playable PCM, each chunk's arrival/sample count, completion/duration/RTF, hashes, native float peak and modeled maximum late supply when starting playout at first chunk. This is NOT audible playback, HF WAN TTFA, H3 first-video latency or Discord continuity. Iterator timestamps alone do not prove causality; combine with audited native source/async decoder path. Nonempty PCM may contain leading silence.

Vox WAV preserves native float32 (no clipping/normalization); Qwen WAV preserves native PCM16 bytes. No resampling, padding, frame/sample drops or waveform repair. Bounds: one utterance≤120s cooperative wall/≤30s audio/≤1024chunks, artifacts≤32MiB. Global provider deadline enforces a hard finite Job even for blocked native calls. Partial output removed/never accepted.

Vox PyTorch allocated/reserved peaks are native-process counters. Qwen client-process counters are NOT server memory. Whole-device NVML observed peak sampled every50ms includes server/decoder/cache/context but can miss transient peaks; label accordingly. Do not compare zero client allocations to native server VRAM.

CPU tests: test_h3_voxcpm2_streaming_candidate.py, test_h3_qwen3_streaming_candidate.py, test_h3_qwen3_streaming_runtime.py, test_h3_tts_streaming_measure.py, test_h3_native_tts_probe_owner.py. GPU execution/quality acceptance remains separate. Server failures retain bounded redacted diagnostics, not reference Base64/secret logging.
