# Current audio path: audit and acceptance

**USER decision: audio solved.** Retain current Comet/OBS/BlackHole route. USER accepts keeping other Comet tabs silent; application isolation/migration is not a completion blocker. Outgoing remote audio and incoming silent-speaker signal were accepted separately below. Shift development priority to video continuation; this does not claim a newly completed causal AV conversation.

No playback, capture, device/routing changes, network observations or GPU execution were performed. Source behavior is not active-runtime verification.

## Scope and user decisions

Discord remains mandatory for sharing with friends. A voice-only Bot is not a replacement for normal-account video calling. Waiting/listening footage is already included in the trained identity LoRA; the open issue is inference control and continuity, not dataset inclusion. USER reports an additional HF allocation equivalent to approximately 2.5 hours of h200x2 and retains budget management. This is not a measured balance, monetary-cap reset or instruction to launch compute during this audit.

## Receive: Discord / Comet → ASR

Current resident chain:
- `research/run_h3_resident_conversation.py:149–167`: endpoint capture then recognized text to Pi.
- `research/h3_personal/resident_endpoint.py:96–105,164–201`: compile/hash-check `stream_h3_comet_pcm.swift`, capture PCM, delimit input and submit to endpoint ASR; require cleanup and finalized coverage.
- `research/stream_h3_comet_pcm.swift:34–52,170–219`: enumerate CoreAudio processes whose bundle equals `ai.perplexity.comet` or starts with `ai.perplexity.comet.`; use `CATapDescription(stereoMixdownOfProcesses:)`, private/nonexclusive, `muteBehavior = .unmuted`; create a private tap-only aggregate with no physical subdevices.
- PCM is converted to 16kHz mono PCM16LE. All matching Comet process audio is included, not a Discord tab/channel/participant filter. Process-set drift fails capture.
- `research/h3_personal/pcm_boundary.py`: energy-based boundary, not semantic end-of-speech.

This is application-process capture, NOT BlackHole input or an acoustic speaker capture. `.unmuted` means the tap itself does not silence the application; it does not prove independence from all downstream/device volume controls. Actual macOS/Comet behavior under the user's volume-key mute remains unmeasured. Discord/tab mute and the actual affected output device also remain unknown.

Older `run_h3_live_voice.py:77–95` uses `/tmp/h3-comet-speech*` and Apple Speech; do not confuse this with the resident endpoint PCM path. That older launcher does not establish the installed binary hash.

## Send: generated AV → Discord

- `research/h3_personal/live_voice_runtime.py:411–423`: intermediate video concatenation uses `-an`, but final mux explicitly adds speech with H264/ALAC and checks decoded PCM equality.
- `research/h3_personal/resident_delivery.py:199–228,251–274`: verify/download/probe AV, then FFmpeg explicitly maps video and audio, copies H264, transcodes audio to AAC128k and sends MPEG-TS to localhost OBS UDP. No physical speaker playback or desktop-audio capture in this stage.
- Expected downstream route: OBS media-source monitoring → BlackHole2ch → Discord microphone.
- Receipt checks (`resident_delivery.py:84–88`) and OBS PID existence do not verify active source/monitor routing or microphone signal. `play_success` establishes FFmpeg completion only.
- OBS Virtual Camera carries video, not this audio. Recording-track checkbox state is not equivalent to monitoring state.
- `prepare_h3_obs_generated.py:20–25` is an old deliberately silent bootstrap, not the later active UDP configuration; resident delivery does not repair OBS configuration.

Encoded UDP sending does not depend on speaker loudness. Active OBS output/source gain, monitor device, BlackHole controls and Discord input remain unverified. Do not conclude that all mute controls are irrelevant.

## Existing alternative worth evaluating

`research/play_h3_virtual_audio.swift:96–125` already binds AVAudioEngine output explicitly to BlackHole and checks device continuity. Resident delivery does not call it. It could remove OBS monitoring from audio delivery, but separate playback timing, AV synchronization, gain verification and failure recovery need evaluation before adoption. It is not an implemented repair or remote-audio pass.

## Follow-up metadata observation

Added `research/inspect_h3_audio_devices.swift`: explicit `--inspect` reads device names/defaults and main/left/right output volume/mute; unsupported properties are null. No audio streams, taps, capture, playback or property writes. Swift6 warnings-as-errors compilation and offline-default execution passed. One explicit metadata inspection reported:
- BlackHole2ch: default input AND default output; main output mute=true, volume_scalar=0.
- Built-in speakers: default system-sounds output; main output mute=true, volume_scalar≈0.125.

This is a strong concrete suspect for missing outgoing audio, not proof of signal attenuation or the historical volume-key target. No controls changed. Both current mute and volume must be considered; merely clearing mute might leave volume at zero. Request explicit approval for a narrowly targeted BlackHole-only change and bounded signal test; keep built-in speakers muted and defaults unchanged.

Local SDK `CATapDescription.h` documents `mutedWhenTapped`: captured process hardware playback is suppressed while a client reads the tap. This is a candidate for silent receive during active capture, not continuous silence between turns and not a proof about device-mute behavior. No capture behavior changed.

Dedicated official Discord desktop app process capture is the preferred isolation candidate over unreliable browser-tab/PID assumptions. It remains application-wide (notifications included), and installation/login/permissions/runtime identification have not been performed.

Direct audio helper cannot be dropped into resident delivery: historical audio-only receipt/WAV contract, 30s limit, no shared scheduled clock or start acknowledgement. Running alongside current muxed OBS audio risks duplicate speech. Keep muxed AV until route diagnosis; any later split requires one audio producer, video-only OBS transport and coordinated presentation timing.

## Approved BlackHole-only repair

After explicit USER approval, added/compiled `research/unmute_h3_blackhole.swift` (Swift6 warnings-as-errors) and ran its explicit `--apply` once. Exact UID target `BlackHole2ch_UID`: output mute true→false, scalar volume0→1. Script verifies built-in speaker mute/volume and default input/output/system-output unchanged. Independent metadata recheck confirms BlackHole unmuted/100%, built-in speaker still muted/≈12.5%, default input/output still BlackHole, system-sounds output still speaker. No playback, capture or GPU execution. This repairs settings only; actual signal/remote audio remain unverified. Default invocation performs no device access or mutation; failure reports possible partial change rather than claiming rollback.

## Remote outgoing audio acceptance

USER approved one existing-audio test without GPU. Verified v4 AV SHA `dd134365205509865a2f44d461cd85606be4869e04d1601689ab0c10dba42993`, H264/ALAC15.5s and existing OBS UDP listener; sent once through unchanged FFmpeg→OBS→BlackHole route (exit0,15.058s). USER confirmed clear reception on iOS Discord: 「ばっちり聞こえました」. Outgoing audio delivery is accepted for this diagnostic with built-in speakers still muted. No new inference/capture or route changes during playback. Receive-path behavior, restart persistence, new causal turns and AV continuity/synchronization are not established by this test. Do not replace the now-working muxed path merely to bypass OBS monitoring.

## Opt-in receive implementation (not activated)

`stream_h3_comet_pcm.swift` now accepts strict `--app comet|discord` and stream-only `--mute-while-capturing` (CoreAudio `.mutedWhenTapped`). Default/legacy behavior remains Comet/unmuted. Discord matches `com.hnc.Discord` or dotted descendants only; actual installed audio helper identities remain unverified. No fallback to Comet. All process drift checks use the same selected app. This isolates applications, not participants/notifications, and does not isolate a Discord browser tab.

`--describe` takes precedence over execution and emits configuration without enumeration/capture; e.g. `--describe --app discord --stream --mute-while-capturing`. Real `--stream` remains permission-gated, raw PCM stdout/max45s; `--check` enumerates only. Muting lasts only during active tap reading, not between turns. No device default changes.

Swift6 compilation and4 offline tests passed (`python3 -m unittest discover -s tests -p test_h3_capture_options.py -v`). Tests execute only describe/argument validation, never capture/enumeration. Existing native capture reused; local reuse searches found no published matching assets. Resident owner still launches legacy Comet `--stream`; these opt-ins are NOT wired into its deployment/authorization contract. No Discord installation/login, capture or playback occurred in this unit.

Next receive acceptance should use an explicitly approved bounded local capture from existing Comet while USER speaks over Discord, retain only aggregate signal metadata (discard PCM, no ASR upload/GPU), and verify nonzero signal with speakers still muted. App migration is a separate operation; do not install/start Discord merely to perform this first check.

## Approved receive signal check

USER approved10s local Comet capture after returning home. Precheck: built-in speaker muted, BlackHole unmuted/volume1; no settings changed. Compiled current Swift6 capture and ran legacy `--stream` with stdout `/dev/null`, stderr aggregate metadata only. Stopped at10s using SIGTERM; wall10.014s,151546frames@16k (~9.47s),105704nonzero samples,peak28072/RMS2574.07. Cleanup=true; exit143 is expected signal termination, not a capture failure; no forced kill. No saved/uploaded PCM, ASR or GPU. Confirms nonzero Comet process audio with speaker muted, not speech identity/content or Discord-only isolation. New muting/app options were not activated. USER subsequently confirmed speaking about audio testing and clarified that only the iOS Discord microphone mute was toggled, not Mac speaker mute. This accepts receive signal under the current silent-speaker configuration; speech content/ASR and application-versus-participant isolation remain unverified.

## Next

1. Read-only inspect current route-selection helpers and available device-control diagnostics; establish which device/control volume keys affected without changing defaults.
2. Evaluate existing direct-BlackHole playback for synchronized AV delivery, preserving current path until evidence favors a change.
3. Investigate Discord-only receive isolation within normal official-client usage; retain Discord and do not substitute voice-only Bot or unofficial account API automation.
4. Obtain per-operation approval before bounded runtime signal tests, capture, playback or routing changes. Observe each hop; no GPU is needed.
