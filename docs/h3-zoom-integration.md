# Zoom route: staged implementation and safety contract

Intent: add Mac Zoom Desktop as an independent AI participant, human Yugo on iPhone, preserve Discord. Use OBS Virtual Camera/BlackHole output and local VoxLock→existing ASR input, not Zoom RTMS/SDK/bot/UI automation. Host/all participants must know it is AI and consent to mixed audio processing/external ASR. No recording/training use.

Source concept: ~/Downloads/H3-You_Zoom_Integration_Concept.md. Actual implementation state/results: `h3-zoom-voxlock-offline-result-20261028-v2.md`; initial slow profile: `h3-zoom-voxlock-offline-result-20261028-v1.md`; control-chain wiring/verification: `h3-zoom-control-chain-wiring-20261028-v1.md`.

## Current versus intended
- Existing production carrier is authenticated paired/continuous HTTP, **not accepted WebRTC**. WebRTC prototypes have finite EOS/quality gaps. Zoom does not technically require transport migration, but concept's selected WebRTC assumption needs explicit reconciliation before live rollout; no silent stack substitution.
- Outgoing OBS/BlackHole unified route unchanged. Adding Zoom means USER chooses/approves same virtual devices in another client; no app/device/routing operation is implied by code implementation.
- Swift capture helper `research/stream_h3_comet_pcm.swift` now accepts explicit --app zoom. Default remains Comet; `us.zoom.xos` +dot descendants is a candidate, not proof all Zoom audio processes are covered. App scope is never participant isolation. Receipt app/bundle/mute metadata is present at selection/stream/stopped.
- --describe is configuration-only and returns before every CoreAudio operation, even with --stream. --check enumerates audio processes and is NOT the same offline guarantee. --stream starts actual process tap/aggregate device and requires per-operation approval. No automatic mute/routing/camera/mic access.

## Local input building blocks
`research/h3_personal/voxlock_streaming.py`: exact hash-pinned v2 FP32 ONNX CPU source/models, stateful16kmono extraction. No automatic model download/fallback. Reference exactly3s float32, selection/conversion caller-owned. Native512window/128hop, h/c/K/V persistent; batch1 default, explicit1..16 batch and1..4 CPU threads. reset abandons tail; terminal flush preserves sample count. Preserve upstream NOTICE for REAL-TSE attribution if distributing weights.

`research/h3_personal/zoom_input.py`:
- EchoGuard starts CLOSED. AI output starts→invalidate epoch/block; output stopped→remain blocked through configured tail. arm requires explicit manual human-turn confirmation, participant consent, finite≤45s deadline; live arm also requires validated echo-tail flag. Offline flag is for file fixtures, not bypassing live safety. initial_idle cannot override active output/tail.
- ZoomPCMReader single owner, selected1024samples/64ms capture quantum. Original capture16kmonoS16→native float extraction→explicit PCM16 conversion. Fixed memory/45s/exact counts/tail, errors/clipping blocked rather than raw fallback/repair. Pending blocked audio discarded; fresh eligible windows need a **new approved capture**, never reused pipe backlog.
- Atomic guard dispatch prevents stale local output release across epoch transition. Already-submitted ASR audio cannot be retroactively withdrawn; caller must abort/reject stale whole turn and final before Pi publication. Capture stop/OS cleanup remains caller-owned.
- `source_contract` is strictly versioned/profile-hashed configuration, NOT actual capture or human-identity proof. Includes candidate bundle, model/source/enrollment/capture hashes, FP32 CPU4/batch8/64ms, explicit limitations.
- `transcribe_extracted` is an injection seam for existing endpoint_asr.transcribe(..., completion_policy='input-finalized'). Validates extracted sample coverage/cleanup/epoch and distinct Zoom origin. It does NOT open capture/resume Endpoint.

## Wired control chain (offline-verified, not live Zoom)
`h3-zoom-control-chain-wiring-20261028-v1.md` records the wiring and its evidence. In short: `research/h3_personal/asr_source_contract.py` holds the two exact immutable variants (legacy Comet unchanged; Zoom adds `input_source` and requires `zoom_capture`) and derives the origin string; `zoom_turn_input.py` supplies the attended pipeline (`AttendedGuard`, TTY `ZOOM-IDLE` confirmation, bounded window, single-use reader, explicit `--stream --app zoom` argv, hash-gated `build_native`); `ResidentEndpoint.capture(..., input_pipeline=)` checks the capture-source pin, prepares the guard before spawning, and attaches `input_source`/`zoom_capture`; `DirectASRInput(..., input_pipeline=)` publishes the expected source and returns `None` on operator decline without closing a healthy lease; the direct launcher rejects `zoom_input` combined with Comet or a non-`ZoomTurnInput`, and the staged spec/worker validate the receipt. A Zoom turn cannot be relabeled Comet and vice versa.

## Clone echo is a fundamental distinction
VoxLock extracts a target voice, not proof the source is human Yugo. Offline original-human and generated-clone fixtures are both preserved. Nonzero output/silence boundary/ASR final is not speaker authorization. Initial PoC is deliberately half-duplex/manual eligible input; no barge-in/full-duplex guarantee. Gate must follow actual outgoing audio lifecycle, not model-generation start/end. Local sink acceptance is NOT remote Zoom audibility; tail needs actual meeting validation, including iPhone acoustic loop. Other-speaker/overlap leakage remains a quality/privacy acceptance gate even with TSE; require all-party consent before external ASR.

## Offline usage (no app/capture/cloud)
```
uv run --project research/voxlock_experiment --locked python -m research.run_h3_zoom_preflight
# Actual LOCAL model/file extraction only; all paths explicit, fresh external result dir:
uv run --project research/voxlock_experiment --locked python -m research.run_h3_zoom_preflight --run-offline --model-dir LOCAL_VERIFIED_ONNX_DIR --reference AUTHORIZED_LOCAL_WAV --input AUTHORIZED_LOCAL_WAV --directory FRESH_PRIVATE_DIRECTORY --seconds 3
```
Reference first3s default, configurable explicit offset0..27; input explicit offset. Single FFmpeg source conversions, not per-chunk resampling. Files stay local; stats/hashes only saved, no raw/output audio persistence/upload. Default does not enumerate devices/models or start ASR. Isolated numpy/ORT lock, no root/HF CLI/iOS environment changes.

Targeted tests (isolated uv project; the root environment has no numpy):
```
uv run --project research/voxlock_experiment --locked python -m unittest discover -s tests -p 'test_h3_zoom*.py'
```
41 PASS: test_h3_zoom_input 10, capture 4, turn_input 5, asr_contract 3, launcher 6, preflight 2, control_chain 1, resident_wiring 10, plus test_h3_voxlock_streaming 8 and legacy test_h3_capture_options 4 outside that pattern; 84 existing-chain regression tests also PASS. Strict Swift6 `-warnings-as-errors` build of `research/stream_h3_comet_pcm.swift` succeeds; only `--describe` was executed. No whole-suite/live meeting acceptance claim.

## Integration / acceptance still required
1. **DONE (offline only):** capture factory/source contract through resident ASR owner, DirectASRInput, immutable expected_asr, local publication and remote worker validation; native finalized coverage refers to the submitted extracted audio, and legacy Comet is preserved. See `h3-zoom-control-chain-wiring-20261028-v1.md`. This is mocked/CPU evidence, not a live Zoom turn.
2. Actual sink/playout→guard wiring, cancellation/remote tail/fresh capture lifecycle and no replay. No source-verified-human assertion.
3. Per-operation approved Zoom process coverage/tap and consented target/non-target/overlap/silence/clone-return tests; assess paced p95/endurance under actual OBS encoding load. Current file fixture RT.59–.61 is not such a test.
4. Separately approve Zoom/OBS/virtual camera/BlackHole/device selection/meeting join. Clearly label AI, verify iPhone and other participant acoustic feedback, actual remote AV and unchanged Discord path.
