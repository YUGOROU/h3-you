# Upstream exact-audio WebRTC migration

## Scope

Latest USER-approved hybrid supersedes the original all-hops plan: keep H3→Mac RTP H264/Opus and add WebRTC for Mac→ASR, but use Network Group authenticated/encrypted direct TCP for Vox Job→H3 Job. See [native TCP guide](h3-network-group-native-tts.md). The Vox DataChannel adapter below is retained experimental work, not the selected deployment path. No public HTTP/WebSocket audio fallback; HTTP remains authentication/text/control/SDP.

**Implementation is partial, not production-selected or WAN-accepted.** Exact RTC carrier, local IPC bridge and model-client seams exist and have actual same-host RTC tests. Remote Endpoint signaling/process ownership, immutable backend selection and top-level ASR/TTS owner integration remain to be implemented. Existing production source paths are not silently switched by these additions.

## Upstream transport

- `webrtc_exact_channel.py`: existing isolated aiortc Session's reliable ordered DataChannel, not Opus. Immutable session/boot/epoch/purpose/byte ceiling handshake. Separate ASR native-event and Vox native-wire purposes.
- Fragment payload≤12,288 bytes, write call≤65,536 bytes, one outstanding outgoing fragment, one unconsumed incoming fragment. Another reader/writer or overflow fails instead of creating a backlog.
- ACK only after the application sink accepts the bytes; it is not an ASR-final, native TTS-EOS, human-speaker, intelligibility or playback proof.
- Each direction has its own count/SHA256/sequence and one verified carrier EOF. Native model EOS remains separate. No padding, resampling, clipping, codec conversion, hidden media route or retry.
- Bidirectional completion barrier before peer teardown. Binding mismatch, duplicate control keys, incorrect sequence/type/hash, premature completion and cancellation are terminal.
- `webrtc_exact_ipc.py`: connected AF_UNIX **stream** socketpairs only. Local IPC separates isolated RTC dependencies from original model and Mac/HF environments. Bounded socket backpressure and half-close; no arbitrary TCP connector. Native consumer closure at verified carrier EOF is recorded explicitly; data-write failures are never suppressed.

## Model seams

### Mac→ASR

`webrtc_upstream_adapter.asr_transport(RTCStream(...))` plugs into existing `endpoint_asr.transcribe(..., _transport=...)`. The existing ASR application-frame parser and native final/commit/sample coverage logic remain intact. Frame syntax is retained as application bytes over DataChannel; there is **no remote WebSocket HTTP Upgrade or WSS connection** in this adapter. The Endpoint RTC adapter still needs to connect this stream to its native ASR event engine. No microphone is accessed by constructing the adapter.

The native engine may retain its private intra-container model API. Any such bridge must be explicit and loopback-only, never a WAN WSS substitute. Current public legacy ASR audio route must be disabled for an immutable RTC deployment before claiming full migration.

### Vox→H3

`RemoteVoxCPMRTC` retains small HTTPS READY/control responses, but all stream bytes come from an explicitly bound RTC-owned AF_UNIX stream. It never calls the inherited HTTPS audio stream. Session/boot/epoch/purpose must match; reserve epoch before uncertain creation; no replay/fallback.

`webrtc_voxcpm_source.stream_native` calls the existing `voxcpm_endpoint_service.Gate.stream` directly into owned IPC. Model pins, native float32 wire framing/hash/counts, serial gate, actual source generation, and independent EOS evidence are unchanged. It is response-only: text is authenticated control, never an uploaded audio prompt. Model `decode()` still requires native protocol END and trailing EOF; carrier EOF alone cannot seal audio.

## Acceptance and next steps

Actual same-host tests prove:

- Exact bytes in both directions, fragmentation, sink-gated credit, carrier hash/EOF and half-close through actual RTC.
- Synthetic native Gate float32 output→RTC→original Vox decoder, with HTTPS media construction forbidden.
- Synthetic PCM→RTC→native ASR application framing and final event→RTC→existing ASR parser, with input commit/full native coverage preserved. This is a fake event engine, not ASR model accuracy or capture acceptance.

Remaining before deployment adoption:

1. Authenticated per-turn Endpoint signaling/lease/identity and isolated child process ownership; short-lived approved TURN after model readiness.
2. Immutable native backend and ASR deployment selectors; disable public legacy audio routes, fail closed on RTC unavailability.
3. Wire factories into finite Endpoint/H3 conversation owners; retain independent source/transport/model-EOS evidence and cleanup.
4. Actual sequential finite cloud verification and separately approved fresh-input/browser E2E. No cloud/device operation was performed for this unit.
