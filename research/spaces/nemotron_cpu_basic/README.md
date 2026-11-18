---
title: H3 isolated Nemotron CPU streaming probe
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: apache-2.0
models:
- nvidia/nemotron-3.5-asr-streaming-0.6b
---

Private, single-stream CPU Basic research probe. NOT a public demo. HF private
Space access control is required; never change visibility to public/protected.
Public test samples only until separate live-call capture/upload approval.

Runtime: official NeMo-Speech.cpp v0.2.0 CPU release. Model: pinned official
Nemotron3.5 Q8_0 (ASR-only experimental precision), NVIDIA Open Model License
(OpenMDW1.1); see upstream model card. Runtime Apache2.0 and bundled notices.
No GPU, microphone invocation, TTS, diarization, audio storage, body/access logs,
auto-scaling, paid hardware upgrade or model substitution.

`/ready` reports engine readiness. Stateful streaming uses WebSocket
`/v1/audio/transcriptions/realtime` (project-specific protocol, NOT generic OpenAI
Realtime): `session.update` before audio with sample_rate16000/language,
binary little-endian PCM16 mono in160ms chunks, then `input_audio_buffer.commit`.
Partials/finals are `conversation.item.input_audio_transcription.delta/completed`.

A required absolute `H3_ASR_EXPIRES_AT` plus maximum1800s foreground server lifetime
prevents indefinite inference/reloads. Container remains inert after expiry;
operator pauses it after the test. CPU Basic zero hourly price, not proof of
realtime. Actual CPU throughput/latency, Japanese accuracy and Comet capture
integration remain unverified until measured. HF outer private authentication
must be checked (anonymous denied) before any audio request.
