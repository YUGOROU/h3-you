# WebRTC delivery / EOS / credit acceptance contract

Status2026-10-29: USER explicitly requests all-media WebRTC and fastest actual talking E2E; warm TTS1.6s is sufficient, no micro-optimization gate. Native producer→isolated RTC sidecar/gateway and browser receiver candidate are now wired; actual local growing IPC→RTP→decoded sink and browser-script contracts passed, but actual H3/browser playback and conversation acceptance remain pending. See native wiring CPU result. Host-only and STUN-only failed, then USER-authorized HF/FastRTC Cloudflare TURN/TLS443 achieved actual HF→Mac ICE/DTLS/control and AV decode. Current result: TURN WAN connection. This is synthetic networking, not native live H3/browser/playback acceptance; no full FastRTC framework migration yet. Historical same-host results: `h3-webrtc-saved-av-and-supply-schedule-result-20261028-v1.md`, `h3-av-closed-loop-analysis-20261028-v1.md`. Do not substitute local RTC after WAN HTTP media or call a saved replay E2E.

## Two explicitly different experimental profiles

|Profile|Transport/content|What can be verified|Current blockers|
|---|---|---|---|
|Standard media|H264/Opus RTP tracks + bounded authenticated control|RTP/decode/count/PTS and receiver sink evidence; source provenance separately|aiortc finite encoder/jitter tail, quality/watermark changes, playout/credit mapping, WAN|
|Exact byte carrier|Reliable ordered DataChannel with original H3AV1 video+PCM, optional separately versioned lossless PCM|Existing original body/region hashes/counts, native paired verification/FIFO/sink ACK|Fragment/message/buffer bounds, source-window reassembly, final control ordering, actual carrier/WAN tests|

DataChannel is real WebRTC but **not standard RTP media-track playback**. Reliable SCTP can still stall on loss/order/retransmission; do not attribute all WebRTC media properties to it. Exact byte transfer is not exact playback. A custom browser/FFmpeg sink must prove its own clock/drain. Standard RTP is the main media exploration branch; an exact-byte branch is not permission to silently substitute it or claim RTP success.

Native H3 reference1024/BF16/nativeFP32/SMD4/124-22/every3/source PCM/watermark/default provider remain unchanged. Do not carry accepted native/Discord quality over to new codec/transport candidates. H264 passthrough or another RTC implementation can be explored, not presumed supported. Using Opus/H264 conversion needs a separately bound candidate profile and USER quality review before adoption.

## State/evidence vocabulary
- **source_validated**: original envelopes/bytes/regions/ancestry/counts/hash, before codec.
- **source_consumed / encoded / buffered**: source/encoder/transport ownership only. No receiver/playback implication.
- **original_bytes_verified**: only when receiver actually verifies original bytes (exact-byte profile). RTP packet counters/metadata cannot establish original PCM/MP4 hashes after transcoding.
- **media_received / decoded**: per-track packet/frame/sample and nominal PTS coverage, not presentation. RTP counts differ from frames; normalized per-track PTS alone does not prove common-clock playout synchronization.
- **sink_accepted**: real ordered receiver-side sink acceptance under its explicit contract. Encoder pipe acceptance is not remote presentation.
- **playout_complete**: receiver-specific actual presentation evidence, not generic `track.recv()`/MSE append/data ACK.
- **session_terminated**: peers/tasks shut down. Can be incomplete/unknown; must never be called media-complete merely because cleanup succeeded.

Existing source cache≤2 is not native two-credit proof. Current saved replay explicitly sets native_producer_credits_exercised/playback_ack_exercised=false.

## Live credit mapping — mandatory gate, not implemented
1. Sole generator owner. Generating + queued/sending/unaccepted windows share exactly2 slots; finite compressed and expanded storage bounds remain distinct.
2. Verification does not release a slot. Only a validated FIFO **receiver sink-accepted ACK received by producer**, after defined real acceptance, can release it. A sender's `bufferedAmount`, RTP send, source-track consumption or received metadata cannot substitute.
3. Exact-byte profile may reuse existing paired wire semantics only after actual original bytes are verified. Standard lossy RTP cannot forge the old `verified` ACK using a source hash it never checked; it needs a separately versioned candidate verification/acceptance contract, not a silent bypass.
4. Control metadata binds finite session/profile/source manifest, window index, global frame/sample ranges and true EOS endpoints. Do not put persona/text/IP/credentials into diagnostic receipts. Authenticate signaling and DTLS fingerprint association before remote use.
5. RTP and separate SCTP control have **no cross-transport ordering guarantee**. Early window/EOS metadata must wait for corresponding media evidence. Final control capacity must remain bounded and cannot deadlock behind unreleased media credits.
6. Final ACK requires successful finite sink finish/drain, not merely final metadata. Missing final ACK/ambiguous failure is terminal, no replay/fallback/queue growth.
7. At window boundaries, use continuous tracks/global timestamps. Withholding old-window decode until next-window RTP arrives must be tested within2 slots; session EOS is a different problem. Never reset clocks/drop/duplicate frames to get a passing counter.

## Finite EOS findings / acceptance
Installed aiortc1.15.0: source EOS does not explicitly flush Opus encoder/resampler; audio receiver prefetch4 retains packets; video jitter buffer waits for a subsequent timestamp to release final frame; RTCP BYE/stop terminates decoder without demonstrated jitter/codec drain. The saved trial's1frame/4160sample difference is consistent with this, despite RTP loss0. Waiting another second or graceful peer close does not establish completion.

**Standard-media current status: incomplete tail.** No tested public aiortc API solves exact EOS under the no-invented-media rule. Do not patch private jitter internals silently. Options require separate evidence:
- Proper reviewed upstream/public encoder+receiver EOS/drain support, or another implementation/browser with demonstrated finite delivery and actual sample/frame coverage.
- Genuine subsequent listening AV can naturally release previous speech tails in an ongoing call, but does not prove exact final session tail. Do not manufacture silence/repeated frames and call it a solution.
- Codec delay/padding and trim metadata, if proposed, must be explicit codec semantics with original source endpoints and quality/watermark tests—not application padding used to hide starvation.

**Exact-byte EOS candidate:** final envelope and payload on same reliable ordered application stream; receiver validates declared lengths/hash/counts/final flag and rejects trailing bytes. Fragment/reassembly and bufferedAmount bounds; final EOS cannot outrun unresolved payload. Existing sink final finish/ACK remains the final acceptance gate. Reliable delivery/peer shutdown alone is insufficient.

Hybrid RTP playback + lossless source copy can verify provenance, not repair RTP tail. Splicing a lossless tail into playback requires a new clock/decoder contract and must not masquerade as ordinary WebRTC interoperability.

## Required validation order
1. CPU contract tests: generation≤1, slots≤2, prefetch≤1, verified ACK non-release, FIFO, finalization before final ACK, finite timeout/failure, no replay.
2. Real same-host carrier/media tests with exact source provenance and bounded memory/cleanup; report source bytes, encoded bytes, decode coverage, source readiness, arrival timing, late/drop/concealment separately.
3. Synthetic-only finite remote signaling/ICE/UDP/WSS check under permitted placement/privacy. New STUN/TURN/relay/provider installation is not implicit.
4. Closed-loop native GPU only after the candidate's source/credit/EOS/quality contracts are tested. Historical fixed readiness and CPU models are not real native acceptance.
5. USER-approved device/playback/OBS/Discord observation last; no automatic reconnect. Finite acceptance is not an unconditional future no-stall guarantee.
