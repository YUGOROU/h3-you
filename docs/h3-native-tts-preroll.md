# Explicit bounded native TTS preroll

## Purpose / scope

USER prefers a bounded initial wait over audible interruptions for a source near RTF1.1. A native source need not sustain immediate equal-rate playback if explicit preroll plus actual H3 conditioning/generation lead covers its finite utterance. Do not treat standalone RTF>1 alone as a coupled-E2E launch blocker.

This policy is implemented in the native PCM bridge and explicit direct conversation native backend; production default remains local complete-WAV. See [backend selection/CPU connection contract](h3-native-tts-direct-backend.md). USER lifted the short headless GPU hold on2026-10-29; first coupled result verifies two-turn PCM/END/native EOS, not gap-free playback or H3-before-source-END overlap. It changes publication timing, not model/precision, waveform samples, playback rate or AV clocks. Local Irodori defaults remain unchanged. Full ASR/Pi/H3/WAN/OBS/speaker acceptance remains separate.

## API

`research.h3_personal.tts_preroll`:
- Frozen `PCMPrerollPolicy(sample_rate=48000, reserve_samples=48000, max_reserve_samples=96000, max_wait_seconds=2.0)`.
- `reserve_samples` is the **threshold** of buffered source audio, not wall delay. Current sample-count gate accepts native48kmono; requested reserve is positive and capped at2seconds. Crossing chunks can overshoot the threshold, bounded by transport chunk size and existing30second source capacity.
- `PCMPrerollGate.observe(available_samples, now, sealed=False)` is observation-driven. It opens once threshold is reached; a shorter source opens only after verified seal. It never publishes empty PCM, synthesizes silence, stretches sound, or grows its target automatically.
- Added wait is measured from first nonempty PCM to admission, separate from cold load/TTFA. Default2second wait cap is checked at observations; **no background timer/hard interruption of blocking I/O**. Remote transport must enforce its absolute request deadline. Late threshold arrival fails closed, never force-starts speech. Once admitted, this cap does not terminate normal subsequent generation.

`relay_native_tts.produce(..., preroll=policy)` opts in. Omitting policy preserves first-PCM publication. Policy is strict immutable input; first publication happens once. Short-source publication occurs after END validation, generator close and ingress seal. Failure cancels ingress and closes remote transport; no retry/replay/fallback. Return `NativePCMReceipt.preroll` records decision, first PCM/admission monotonic times, wait, requested reserve and short-EOS path.

Existing bridge conversion remains explicit: native float32le→S16LE48k, clip/round-to-nearest-ties-even/saturate. Native float and converted PCM retain separate hashes/counts/saturation receipt. Preroll adds no conversion, sample drops/padding or watermark claim. Protocol END is not independent model EOS.

## H3 handoff

Reserve gates attachment/publication; `StreamingAdmission` independently needs its native waveform window plus validated resampler lookahead before speech can begin. These conditions are combined by readiness, **not by appending another reserve after the H3 window is ready**. Continue listening while pending; after speech begins, starvation remains explicit wait, not silent/listening replacement. Existing common clocks/persistent decoder/tail/credits remain unchanged.

`tests/test_h3_native_preroll_handoff.py` exercises real CPU admission/ingress APIs with fixed124/22 geometry and the pinned context contract, demonstrating the thresholds are max-like, not summed. It does not execute the native resampler/H3 model or prove video supply.

A1second threshold is an initial candidate based on the retained six source traces, not a guarantee for30second utterances, cold starts, load spikes, WAN or all texts. Do not automatically enlarge it. Report residual deficit and actual output gaps before changing the finite policy.

## Saved-source verification

```
python3 -m research.run_h3_tts_preroll_preflight \
  --source-result PRIVATE_EAGER_OUTPUTS/result.json \
  --verify-native-wavs --output FRESH_PRIVATE_RESULT.json
```

Offline only: no cloud/network/device/playback/GPU operation. Evaluates0.5/0.8/1second thresholds against existing measured arrivals. Hash/format/count-checks original saved WAV, replays actual native bytes through the bridge using a virtual clock, and validates the publication time/prefix against the causal gate. No future arrivals are used to choose admission. Whole traces are used afterward to compute residual supply lateness, which is **not actual sink/playback evidence**.

See initial result.
