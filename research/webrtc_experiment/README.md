# Isolated uv WebRTC experiment

This is a separate uv project, not a dependency change to the root project, HF CLI, iOS tools or H3 model environment. Python3.12, aiortc1.15.0 and transitive dependencies are fixed by `uv.lock`. USER approved this isolated installation on2026-10-28. Library/bundled-codec redistribution licenses require review before packaging/deployment; an aiortc BSD license alone does not establish every bundled codec's terms.

From repository root:

```sh
uv sync --project research/webrtc_experiment --locked --python .venv/bin/python
# Offline availability report: no peer creation or network
uv run --project research/webrtc_experiment --locked python -m research.run_h3_webrtc_probe
# Explicit finite same-host synthetic audio/video/control (no capture/playback)
uv run --project research/webrtc_experiment --locked python -m research.run_h3_webrtc_probe --run --duration 4
# Contract + actual optional-library peer/cancellation/codec/tail regressions
uv run --project research/webrtc_experiment --locked python -m unittest discover -s tests -p test_h3_webrtc_synthetic_probe.py -v
```

`--run` creates one child with a30s parent hard timeout; async operation20s/peer cleanup5s. It sends synthetic440Hz PCM tone and changing grayscale252×448 video, consumes received frames headlessly, and negotiates only Opus and first supported H264 capability (no codec fallback). Original mono PCM is encoded as Opus; received audio may be stereo/lossy. No private H3 media, output speaker, camera, microphone, screen, GPU or cloud launch. No public listener, external signaling/STUN/TURN. Host candidates are gathered on local interfaces, so this is same-host, **not verified loopback-only**. SDP/candidate/IP/credentials are never printed. Sources stop without extra silence or frames to flush tails. This is not complete-source-delivery, WAN, quality, lip-sync or actual playout acceptance.

Counters distinguish RTP packets from decoded frames/samples. `jitter_rtp_timestamp_units` is not seconds (Opus48000/H26490000clock). Receive gaps are arrival-to-consumer observations, not audible stalls. Source and receive first-PTS may independently normalize; don't use them alone as proof of RTCP/AV playout synchronization. Explicit source count differences and `sample_exact=false`, `source_tail_flushed=false`, `playout_verified=false`, `wan_verified=false` remain part of the result.

## Saved H3 source replay (explicit same-host experiment)

```sh
# All source available at start, but lazy source cache never exceeds two windows
uv run --project research/webrtc_experiment --locked python -m research.run_h3_webrtc_probe --run --capture-dir PRIVATE_CAPTURE_DIRECTORY
# Frozen previous packet-ready availability, not a live producer/credit loop
uv run --project research/webrtc_experiment --locked python -m research.run_h3_webrtc_probe --run --capture-dir PRIVATE_CAPTURE_DIRECTORY --producer-timing PRIVATE_PRODUCER_TIMING_JSON
uv run --project research/webrtc_experiment --locked python -m unittest discover -s tests -p test_h3_webrtc_saved_source.py -q
```

Saved replay validates the full source before peer setup (≤32MiB capture), rechecks immutable body hashes while loading, derives frame/sample PTS from global indices, and keeps tracks continuous across all windows/roles. No clipping via `--duration`. A final partial audio frame is not padded. Only two compressed source windows are held; local native codec buffers remain opaque. Source consumption is NOT a network/sink/playout ACK and does not exercise the native two-credit protocol. H264/Opus reencode and adaptive behavior are explicit quality changes, not accepted defaults.

The saved-mode child has a separate100s hard deadline (operation90s/total cleanup5s); it does not extend a live lease. Track source async generators are explicitly closed after sender shutdown. Timing offsets require same-origin producer clock, indices and video/PCM sizes; legacy timing is not cryptographically bound to source, which is explicitly reported. Late sources catch up with original timestamps/no drops; decoded arrival gaps are not actual playback gaps. Session tail delivery remains incomplete, so these receipts never prove exact source delivery or production readiness.

## Offline conditional flow model (root Python, no WebRTC execution)

```sh
.venv/bin/python -m research.run_h3_flow_model --capture-dir PRIVATE_CAPTURE_DIRECTORY --producer-timing PRIVATE_PRODUCER_JSON --receiver-timing PRIVATE_PUMP_JSON --sink-timing PRIVATE_UNIFIED_TIMING_JSON
.venv/bin/python -m unittest discover -s tests -p test_h3_av_flow_model.py -q
```

This recomputes two-credit/ACK feedback from observed pull/sink durations. It does not replay historical ready times or credit waits. Forward/reverse splits and hypothetical decoded-window consumption are explicitly conditional, not WAN, GPU or playback measurements. Original capture is hash-validated but not decoded. Finalization uses observed receiver accepted-to-ACK-enqueued duration as a proxy, not isolated finish or a solution for RTP tails. Candidate EOS/ACK design: `docs/h3-webrtc-delivery-contract.md`.

## Live connection path (2026-10-29)

USER now prioritizes first actual talking E2E, all-media RTC, rather than more source benchmarks. `webrtc_live_source` accepts newly arriving original paired windows (not saved-file enumeration); `webrtc_live_peer.Session` exchanges external offer/answer with H264/Opus, bounded control, host-only default plus explicit approved Cloudflare STUN/HF-TURN profiles; `webrtc_signal_server.SignalServer` provides authenticated finite HTTP signaling. These are not yet native producer/receiver-sink credit integration. Source consumption is never remote playback acceptance.

```
uv run --project research/webrtc_experiment --locked python -m unittest discover -s tests -p 'test_h3_webrtc_live*.py' -q
uv run --project research/webrtc_experiment --locked python -m unittest discover -s tests -p 'test_h3_webrtc_signal_server.py' -q
uv run --project research/webrtc_experiment --locked python -m unittest discover -s tests -p 'test_h3_webrtc_connect_owner.py' -q
# No launch by default (existing HF CLI Python, no installs):
~/.local/share/uv/tools/hf/bin/python -m research.run_h3_webrtc_connect
```

Explicit `--run --directory FRESH_PRIVATE_ROOT` owns one CPU-only HF Job600s/attempts1 plus a70s isolated local RTC child; no models/private media/devices. It never creates a second Job automatically or replays unknown creation. Authenticated `/status` gives session/per-process boot/expiry, `/offer` consumes one bounded offer. SDP is exchanged only in RAM over authenticated TLS, never stored/logged. Candidate receipts contain counts only, not addresses. HTTP is signaling only; no media fallback. Existing source/uv hashes are verified; source ZIP is an ordinary private Git blob, not a dependency install into HF/Mac root environments.

Host-only and then authorized STUN-only failed. USER subsequently approved HF-provided Cloudflare TURN using the official FastRTC service; actual HF→Mac ICE/DTLS/control/RTP decode succeeded with `--hf-turn` (TLS443-only). Result: `docs/h3-webrtc-turn-result-20261029-v1.md`; CPUJob terminated, activeJobs[], no GPU. No full FastRTC installation/model replacement: reuse its verified credential API protocol first. Short-lived credentials only in memory/Job secrets/local child environment, never source/config/argv/logs. Host-only stays default; `--cloudflare-stun` and `--hf-turn` are mutually exclusive explicit profiles. No other provider fallback. `webrtc_turn.py` validates the fixed broker/returned URLs and selects only turns:turn.cloudflare.com:443?transport=tcp. Mint near peer startup rather than burn600sTTL during H3 cold load. This is synthetic WAN reception, not native H3/browser/playback/E2E acceptance; finite codec tail remains unverified. Next actual live model→browser/input integration, no repeated network matrix. Discord/device/call operations still require notification/approval.

Results/gates: `docs/h3-av-closed-loop-analysis-20261028-v1.md`, `docs/h3-webrtc-saved-av-and-supply-schedule-result-20261028-v1.md`, earlier `docs/h3-webrtc-synthetic-uv-result-20261028-v1.md`, and CURRENT.md. Optional real-peer tests skip without aiortc/PyAV; skipped tests are not passed WebRTC experiments.
