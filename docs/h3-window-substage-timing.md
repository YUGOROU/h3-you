# Window-indexed worker timing

Metadata-only diagnostic instrumentation. No precision/model/reference/PCM/decoder/clock/credit changes and no extra CUDA synchronization. Implemented in window_profiler.py, LiveConversation.step, LiveNativeBackend.__call__, and run_worker.

## Evidence
- `h3_window_substages`: one emitted snapshot per window, index/role and allowlisted stage start/end/duration/status.
- `h3_worker_activity`: complete-WAV TTS / canonical PCM activity intervals.
- `h3_window_profile_end`: bounded window count/drop counter.
- All above share **one worker-local monotonic origin**. Correlate with each other, not absolute Mac/pump/GPU-rank clock origins. Compare other timers by durations/index only.

Stages:
- driver_total, driver_prepare, driver_backend, driver_complete.
- backend_total, role_switch, manifest_write, native_call, native_generate_sync, evidence_read, evidence_validate, ancestry_read, video_export.

`native_generate_sync` is the existing returned parent-runtime interval around generator.generate + existing CUDA synchronize. `native_call` includes request/quality checks. Neither is transformer-only, per-rank CUDA event timing, or a split of native conditioning/denoise/learned VAE. `backend_total` excludes lock acquisition and initial manifest validation; driver_backend includes these. driver_complete includes packet composition/source cleanup/emitter, not validation alone.

Complete-WAV TTS runs concurrently and shares the profiler clock. Interval overlap is **correlation**, not causal GPU contention proof. It includes child synthesis/watermark/result overhead; it is not isolated TTS kernel time. Incremental sentence synthesis/continuous resampling do not yet have complete shared-origin activity instrumentation.

## Bounds / failure semantics
- ≤144 windows, allowlisted stage keys, ≤16 activities, elapsed≤1800s, defensive snapshots.
- No prompt/text/images/audio/paths/source IDs/hashes/raw errors in new events.
- span/record/emitter errors are best-effort drops and do not mask body exceptions. Failed native operations retain existing quarantine/no-replay semantics.
- Repeated incremental WAIT prepare/total attempts retain latest attempt, attempts count and prior execution-duration sum. Intervening wait/scheduling gaps are not included in that latest duration. Backend stages never silently replay.
- Rows emit after successful packet completion; remaining partial rows emit during worker cleanup. Abrupt provider/process cancellation may lose current/last events. Drop count/absent events must remain visible.

**Known limits:** stdout write/flush remains synchronous, so a blocked logging pipe can affect timing/latency despite caught errors. Old PhaseTiming emission also retains pre-existing primary-error masking risk on output failure. Do not claim zero instrumentation overhead or universal noninterference from CPU tests. Parent timing does not show native substage cause or actual receiver/playback continuity.

## Tests / use
```
~/.local/share/uv/tools/hf/bin/python -m unittest discover -s tests -p test_h3_window_profiler.py -v
```
Real worker attaches one profiler to the existing driver/backend, preserving their public signatures and other callers' absent-profiler behavior. This is measurement, not an alternate native backend. CPU fake-backend equality proves contract preservation, not GPU quality/speed acceptance.

Direct deployment copies only first-party Python sources, excluding nested uv .venv/hidden caches/site-packages/venv and refusing first-party file symlinks. The uv WebRTC environment is not uploaded to HF by this path.
