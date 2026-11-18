# Conversation VAD / native interruption boundary

ASR WSS is the USER-approved exception to HTTPS restoration. This guide describes the integrated device-free HTTPS entrypoint and combined sink-stop/re-entry evidence, **not accepted human speech, device audibility or conversational latency**.

## Comet input boundary

`ResidentEndpoint.capture` now wraps the existing approved Comet stream with `local_vad.VADBoundary`. `DirectASRInput` passes `planned_start=None` for immediate input; `None` is supported explicitly by the existing capture-time guard. There is no ordinary three-second countdown or ten-second minimum. Zoom's separate approved/manual gate remains unchanged.

No new model, install or dependency is required. The acoustic candidate uses normalized energy, zero crossings and periodicity to produce a **heuristic likelihood score**, not a trained/calibrated posterior. Synthetic tests do not establish human speech quality. Sine/AI voice can be speech-like; this is not identity or echo cancellation.

`VADConfig` centralizes the initial policy:

| Parameter | Default | Purpose |
|---|---:|---|
| frame | 20 ms / 16 kHz mono PCM16 LE | bounded analysis |
| onset | 60 ms | establish candidate with score ≥0.65 |
| minimum speech | 120 ms | reject brief candidates before publication |
| offset score | 0.35 | preserve weak continuation after onset |
| quiet | 500 ms (allowed 300–800) | end application utterance, not native ASR final |
| pre-roll | 200 ms before candidate onset | retain word beginning |
| no speech | 8 s | reject a silent/noisy window without stopping ASR lease |
| maximum | 45 s from window start | finite capture, not 45 s after onset |

The retained confirmation buffer is at most pre-roll + minimum speech (320 ms at defaults), plus bounded read output. It never contains blocked-period input. Before confirmed onset, older authorized input is discarded locally; after onset, bytes are streamed in original order without resampling/padding/replay. VAD times include sample offsets in the captured window and local monotonic observations, not remote clock differences or physical-device speech latency.

`endpoint_asr.transcribe` still requires genuine finalized coverage for speech. A zero-byte VAD result closes transport without sending an empty commit or inventing a final transcript. `ResidentEndpoint` treats that explicitly clean no-input result as `NoSpeech` and preserves the healthy lease.

Capture stop occurs through `on_input_stop` before joining a blocked sender. `ResidentEndpoint` serializes spawn/stop and prevents spawning after cancellation. The hook must be bounded/cooperative; VAD sample-count limits cannot preempt arbitrary blocking stream reads. Existing capture termination/reaping remains the owner's responsibility.

## Turn epoch guard

`turn_epoch.TurnEpoch` provides finite, thread-safe monotonic turn admission with LISTENING/CAPTURING/THINKING/SPEAKING/INTERRUPTING/DRAINING/CLOSED. Onset invalidates older work immediately in memory; commit can be reserved once; EOS, interruption, failure and session stop are distinct. No callbacks or IO run under its lock.

`turn_control.TurnController` composes the guard with exclusive interrupt-enabled `Commands` and a matching device-free `EpochAVSink`. `begin_input()` invalidates the old sink before waiting for control IO; `commit_input()` reserves finalized input once; `publish_reply()` quarantines stale Pi callbacks. A publication already in flight may finish once, followed by an ordered interruption, never a replay. The existing direct entrypoint attaches it in explicit `--headless-interrupt` mode. It does not attest ASR provenance or automatically attach VAD/capture. A point-in-time `is_current()` check alone cannot revoke buffered output.

## Native safe-boundary transition

`LiveConversation.request_speech_interrupt(epoch)` records a request. If a native window is reserved, it must commit and receive decoded ACK once before `retire_interrupted_speech()` can proceed. No CUDA exception, forward retry, model reload, final flush or PTS rewind is used.

For unfinished speech, retirement attaches an explicit marker to the next immutable native window:

```json
{"previous_speech_interrupted": {
  "schema": "h3-speech-interruption-v1",
  "source_id": "previous-logical-source-id",
  "committed_frame": 226
}}
```

The marker belongs inside `streaming_audio`. It must match the prior speech ID and current global committed frame, have a new logical source at index zero, and cannot occur on the first window. Already-final speech and unstarted pending speech do not need this marker. Interrupted sources cannot be reused.

Both-rank allocation applies the marker before `StreamingWindowOwner.start_window`. `interrupt_source` requires a healthy, complete immutable native window with decoded ACK, no active token and the existing five-frame decoder lookahead. Interruption is stored separately from natural source finality; `source_final` is never fabricated. Visual/audio history, persistent decoder, owner counters and original quality remain intact.

**Important:** old ingress/spans survive for the five-frame decoder tail. The explicit worker interruption path generates one real listening quarantine window before attaching fresh speech. Both the interrupted packet and old-tail/listening packet keep the OLD presentation generation and are discarded as paired units. Fresh speech starts afterward, so its beginning is not discarded. This conservative policy adds a full listening window of latency; it is not a mandatory lower bound for every possible cut-boundary design.

## Explicit worker and transport APIs

Enable on both ends only:

```python
commands = Commands(transport, session, turn_interruptions=True)
# Borrow the existing warm model/TTS; select HTTPS paired transport explicitly.
run_worker(..., media_transport='http-paired-v1', turn_interruptions=True)
commands.send('interrupt_turn', old_turn_epoch, {
    'schema': 'h3-turn-interruption-v1', 'presentation_epoch': new_generation})
```

`Commands.send` serializes the independent input/control lane. The worker retains its command reader, old TTS drains to verified END, stale publications are quarantined, and native forward completes once. Never prematurely close a Vox stream: that poisons its serial Gate/client. Two-turn/queue/lease bounds remain unchanged.

`AVEnvelope.presentation_epoch` is independent of fixed `LiveAVBinding.epoch`. Canonical wire serialization omits default zero; explicit generations are positive integers and bound into ACK identity. Use `paired_av_wire.envelope_record()`, not plain `dataclasses.asdict()` when constructing wire JSON.

`Receiver.discard_interrupted(lease, certain=True)` / `Sender.discard_ack(..., certainty=True)` release verified FIFO credits using wire stage `discarded-interrupted`, not `sink-accepted`. Sequence, hashes and global clock still advance. `complete`/`transport_complete` mean final transport settlement; `presentation_complete` is false after any discard. ACK is still not acoustic proof.

`EpochAVSink` wraps existing headless FFmpeg encoding, with reusable paired invalidation and no device output. Invalidation publishes a new floor first, then kills/reaps both old output tracks. It prevents late old-epoch process submission and checks stale source tails. New private NUT spans have local timestamps **with explicit global index/frame origins**; transport/native PTS never rewind, and gaps are recorded as discontinuities. `ReceiverPump` supports the explicit sink disposition with and without prefetch. Do not substitute this implementation for OBS/browser/device flushing.

With `observe_output=True`, the same sink follows its actual growing NUT span through FFmpeg PCM decode. It accounts every stdout sample, including buffered PCM drained after kill, then confirms EOF. Stop criteria use that output cessation bound; response criteria use the first actual decoded new-source PCM and its full sample/hash match. Never substitute selector time, encoded-file write, transport ACK or encoder input for this measurement. This is still not a speaker/device measurement.

## Existing Endpoint HTTPS owner

`voxcpm_https_lease.lease(fresh_private_directory, parent_campaign=...)` verifies the exact independently paused parent, publishes a fresh immutable runtime, PUTs/resumes **the existing** `h3-voxcpm-l4-a474fd64f740`, validates native READY, and independently pauses it on exit. No new Endpoint collection POST. Preserve the outer finite operation alarm. After restoration, supply its new lease directory explicitly as parent; the old RTC default intentionally refuses deployment drift.

## One existing entrypoint

```sh
python -m research.launch_h3_direct_conversation \
  --directory <fresh-private-campaign> --headless-interrupt \
  --vox-parent <reviewed-paused-parent>
```

Without `--run` this is offline description only. With `--run`, the existing CLI reads exactly two new typed turns, leases the existing Vox Endpoint, selects the immutable interrupt-enabled worker/READY and continuous HTTPS, and runs real observed headless output. It rejects capture/OBS/BlackHole/RTC/alternate TTS/endurance combinations before resource startup. Same owner finite alarm/expiry/no-replay/pins/terminal confirmation apply. No new runner or framework.

The first actual source PCM output is followed by a 200ms programmatic interruption; this is a device-free scenario, not VAD/human input. `stop_criteria` and `response_criteria` are separate. The former has a 200ms initial target on decoded stdout EOF; the latter reports functional re-entry separately from unaccepted product latency. Native new-source PCM must match the entire decoded sink output source count/hash.

Optional `--visual-intents` on this SAME entrypoint binds natural-language Pi camera/action intent through the versioned command and both-rank native conditioning/cache path. It explicitly selects latest-tail rather than the legacy headless every3 anchor policy, and retains the same native conditioner under finite ownership; no new model/runner/renderer. Content achievement remains unverified. See `h3-pi-director.md` and `h3-visual-intent-native-wiring-result-20261031-v2.md`. Explicit `--review-av` now binds paired cut mode: preserve the quarantine forward, derive its old prefix from verified native commit/emit, full-decode/verify first, submit only the paired suffix with honest partial disposition. Save private NUT/MOV/reference sidecars under2GiB; no playback/upload. Actual latest review run kept97frames that v2 discarded; source2fullPCM matched, new decoded response19.863s (not faster acceptance). Known late-before-step EOS/tail race may conservatively miss salvage; no unsafe upgrade. See `h3-visual-review-and-latency-result-20261031-v3.md`; saved files are for USER visual judgment.

Latest validated parent: `P/visual-review-20261101-093915-EF24406B/vox` (P=`$H3_PRIVATE_ROOT`). Older component-probe commands remain historical reproduction code, not the next workflow. Do not stack new standalone GPU checks.

## Tail/cut review and remaining acceptance

Five native lookahead frames and the existing conditioning history must survive. A complete 102-frame generated quarantine window is a conservative presentation policy, not an inherent required amount. A future **paired prefix cut** can decode/verify the whole native packet, omit just the interrupted prefix at the committed global boundary, then present its suffix with matching video/audio timestamps. This must explicitly account partial disposition/omitted ranges rather than fake full sink acceptance, and preserve new speech offset0/all PCM. No native reset, rate change, padding, quality or renderer substitution. Source readiness may still require genuine listening windows, so a full 4.1s response reduction is not guaranteed. See the current result for concrete measured windows and conditional savings.

- Combined current Vox→interruptible H3→continuous HTTPS→real headless sink: accepted at the single typed device-free scenario level. Stop EOF bound1.242ms, new PCM12.771s, new source276,480samples exact. This is not low-latency conversation acceptance.
- Actual ASR deployment still needs explicit legacy WSS restoration; ASR was not resumed.
- Input tap/echo safety and human VAD behavior are unresolved. Synthetic success cannot settle them.
- **No real mic/Comet capture, real output/browser/OBS or Discord operation is authorized by this result; latest USER hold remains.**
- Latest saved-review/cut result: `h3-visual-review-and-latency-result-20261031-v3.md`:97frames retained via partial disposition, review MOV/NUT/reference files saved, stdout EOF2.101ms/newPCM19.863s, visual content and speed unaccepted. Prior v2 native-wiring result: `h3-visual-intent-native-wiring-result-20261031-v2.md` (EOF1.231ms/newPCM19.849s). It is not a same-condition speed comparison with the nonvisual v5 result `h3-conversation-single-entrypoint-result-20261031-v5.md`. Historical component GPU evidence: `h3-conversation-device-free-gpu-result-20261031-v4.md`.
