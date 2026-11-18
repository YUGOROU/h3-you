# Private CPU Basic streaming ASR Space

## Result: 2026-10-19

USER approved deploying a private CPU Basic Space, without GPU or paid Upgrade.
Created `YUGOROU/h3-nemotron-cpu-basic-096f00b0dd`. **Now PAUSED**, API-confirmed.
CPU Basic2vCPU/16GB, zero hourly hardware price. No paid CPU hardware requested,
no local model inference, no Discord capture or personal audio upload.

Actual public-file streaming test:
- HF private access checked: anonymous `/ready`404; authenticated ready reports
  `device:cpu`, `capabilities:[asr]`. CPU Basic actual hardware confirmed.
- Public pinned upstream JFK WAV,11s/16kHz monoPCM16, sent at realtime speed in
  160ms binary WebSocket frames, plus1s silence, then commit.
- First nonempty partial **1.842s** after WebSocket connection.
- First utterance-segment final **4.974s**; not the whole-file final.
- Input finished at11.843s; last speech-segment final **14.774s**.
- Last final arrived **2.931s after input ended**. Four final segments reproduce
  the sample's spoken words; no Japanese quality/CER measurement.
- First partial arrived while input was still being sent: actual incremental
  input/partial-result streaming demonstrated, not whole-file REST or SSE.
- `audio_processed` progress implies observed wall/audio ratio **1.283** across
  the reported progress span. This includes transport/scheduling, not isolated
  kernel RTF. Progress lag grew; this configuration does NOT pass sustained
  realtime throughput. One short English trial is not a universal Basic failure.
- No protocol errors; Space paused in client `finally`, then PAUSED checked again.

Basic is a successful functional proof but not yet a sustained realtime solution.
Next compare CPU tuning/Upgrade only under explicit operational approval, and
measure Japanese latency/accuracy on public samples before real Comet input.
A later8vCPU32GB Upgrade costs$0.03/hr (30min$0.015); it is NOT currently enabled.
Upstream ggml runtime's default CPU thread setting is4; `OMP_NUM_THREADS=2` alone
is not proof of limiting ggml to2. HTTP `--threads2` means HTTP workers, not CPU
compute threads. Investigate actual thread controls before attributing all delay
to Basic hardware or promising an Upgrade speedup.

## Artifacts and provenance

Project:
- `research/spaces/nemotron_cpu_basic/{Dockerfile,start.sh,README.md}`
- `research/launch_h3_asr_cpu_space.py`: offline staged plan by default;
  `--deploy` explicitly creates private CPU Basic only; `--status`/`--pause`.
- `research/probe_h3_asr_cpu_space.py`: public-sample-only paced protocol probe,
  offline by default, `--run` explicit; pauses in `finally` unless explicitly
  `--keep-running`. Do not use keep-running as an automatic renewal.
- `tests/test_h3_asr_cpu_space.py`, `tests/test_h3_asr_cpu_protocol.py`:7 CPU tests,
  including staging tamper/visibility/hardware checks, frame bounds/masking and
  correct distinction between first-segment and last final timing.

Private receipts under
`$H3_PRIVATE_ROOT/`:
- `cpu-space-nemotron-20261019-v1`: first deployment, terminal startup argument
  error (`serve: unexpected argument: false`). No model inference. Caught because
  app remained APP_STARTING; no silent paid upgrade or blind replay.
- `cpu-space-nemotron-20261019-v2`: corrected immutable snapshot and
  `public-stream-result.json` with all public transcript events/timings.
- Corrected commit `7d5ecbe0e8ecc35024c56e93a8c491e9c562bc8a`, parent
  `0c5ba3cbb7da5af0d663a692856557825371c7a4`. Boolean override syntax must be
  `--asr.batching.enabled=false` / `--asr.endpointing.enable=true`, not detached
  `false`/`true` words. Added regression assertions. Original expiry preserved.

Pins:
- Debianbookworm-slim image digest
  `sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587`.
- Official NeMo-Speech.cppv0.2.0Linuxx86_64CPU archive SHA256
  `f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507`.
- Nemotron3.5ASR Q8_0, revision`1c8deaecc64b91f034d73e08dd8b64625eb3395d`,
  741548352bytes/SHA256
  `a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae`.
  ASR-only explicitly experimental precision, not H3/nativeTTS quality changes.
  Model OpenMDW1.1; runtime Apache2.0 with bundled third-party notices.

## Runtime/security/lifecycle

Container packages only; no Mac/Homebrew/dependency changes. Non-root user,
explicit CPU device, one model, batching off/state arena1, HTTP workers2,
request/audio cumulative bound2MB, noUI/access logging/microphone invocation,
no TTS/diarization/autoscaling. Prebuilt CLI includes capture capability, but
neither entrypoint nor probe uses `--live` or any physical capture API.

Private HF gateway is the authentication boundary, verified before audio.
No broad HF credential in image, argv, Space secrets, body logs or model URLs.
Control/client reads cached token in memory; only HF hosts receive it. Never
change visibility to public/protected. A separate runtime API key is not set,
so anonymous-denied verification is required on every probe/deployment.

Server uses absolute `H3_ASR_EXPIRES_AT=1792396210` plus maximum1800s runtime;
no automatic extension. On timeout/error it stays inert rather than repeatedly
reloading models. Operator/client must still pause. Restarting an old paused
snapshot after its deadline will NOT reload the model; a new explicitly approved
finite snapshot/deadline is required. No keepalive/background watcher.

WebSocket protocol is upstream project-specific:
`/v1/audio/transcriptions/realtime`, initial `session.update`, PCM16 input,
partials/finals, commit. **Its delta events are incremental text fragments**
(e.g. `And` then ` so`), unlike Together's documented replacement-style deltas.
Future Comet/Pi adapter must append them per utterance and respect finals,
revision/stale suppression and exact source labeling. Not yet integrated into
`run_h3_live_voice` or continuous conversation controller. No new causal AV E2E,
Japanese acceptance, endpoint safety or barge-in claim.

See `docs/h3-asr-provider-verification.md` for REST alternatives and Together402,
and `docs/h3-compute-budget.md` for budget. CPU Basic added$0 hardware spend;
HF conservative bound210.830037/remaining5.783300 remains unchanged.
