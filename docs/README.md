# docs

One file per topic. These are the notes needed to run and verify the pipeline, not a chronological record.

## Start here

| Document | What it covers |
| --- | --- |
| [h3-you.md](h3-you.md) | The research framing: what the project is trying to show |
| [personalized-h3-runbook.md](personalized-h3-runbook.md) | Preprocessing, training gates, and the end-to-end path |
| [h3-pi-director.md](h3-pi-director.md) | The text-only conversation director and its two tools |
| [h3-conversation-vad-and-interrupt-boundary.md](h3-conversation-vad-and-interrupt-boundary.md) | VAD, turn boundaries, and interruption semantics |
| [ios-camera.md](ios-camera.md) | Building, installing, and using the iPhone capture app |

## Generation

| Document | What it covers |
| --- | --- |
| [h3-fast-ref2va-smd.md](h3-fast-ref2va-smd.md) | The fast Ref2VA + identity candidate |
| [h3-fastvideo-ref2va.md](h3-fastvideo-ref2va.md) | The native FastVideo Ref2VA path |
| [h3-realtime-first-experiment.md](h3-realtime-first-experiment.md) | The speed-first experiment, kept separate from native acceptance |
| [h3-acceleration-experiment.md](h3-acceleration-experiment.md) | Precision and decoder acceleration, in its own runtime |
| [h3-ordered-reference-conditioning.md](h3-ordered-reference-conditioning.md) | Ordering a person and their surroundings into one condition |
| [h3-reference-geometry.md](h3-reference-geometry.md) | Reference canvas geometry and what may be cropped |
| [h3-quality-cpu.md](h3-quality-cpu.md) | CPU-side image quality checks |

## Identity

| Document | What it covers |
| --- | --- |
| [h3-identity-pipeline.md](h3-identity-pipeline.md) | The identity pipeline end to end |
| [h3-identity-model-lock.md](h3-identity-model-lock.md) | Pinned model revisions |
| [h3-identity-guidance.md](h3-identity-guidance.md) | The experimental guidance-target correction |
| [h3-identity-loss-audit.md](h3-identity-loss-audit.md) | Where identity is lost across the training stages |
| [h3-component-cache.md](h3-component-cache.md) | Component reuse and what counts as evidence of it |

## Speech

| Document | What it covers |
| --- | --- |
| [h3-native-tts-direct-backend.md](h3-native-tts-direct-backend.md) | Native streaming TTS |
| [h3-native-tts-preroll.md](h3-native-tts-preroll.md) | Bounded native TTS preroll |
| [h3-native-streaming-tts-probe.md](h3-native-streaming-tts-probe.md) | The native streaming probes and their contracts |
| [h3-network-group-native-tts.md](h3-network-group-native-tts.md) | Native Vox Job to H3 Job over Network Groups |
| [h3-irodori-inference-endpoint.md](h3-irodori-inference-endpoint.md) | The Irodori TTS Endpoint |
| [h3-asr-inference-endpoint.md](h3-asr-inference-endpoint.md) | The ASR Endpoint |
| [h3-asr-cpu-space.md](h3-asr-cpu-space.md) | ASR on a CPU Space |
| [h3-audio-path-audit.md](h3-audio-path-audit.md) | The audio path, device by device |

## Transport and delivery

| Document | What it covers |
| --- | --- |
| [h3-webrtc-delivery-contract.md](h3-webrtc-delivery-contract.md) | What counts as a delivered frame or sample |
| [h3-upstream-webrtc.md](h3-upstream-webrtc.md) | The upstream WebRTC path |
| [h3-window-substage-timing.md](h3-window-substage-timing.md) | Window timing semantics, and what they do not prove |
| [h3-incremental-av-handoff-design.md](h3-incremental-av-handoff-design.md) | Incremental audio/video handoff |

## Infrastructure

| Document | What it covers |
| --- | --- |
| [h3-modal-runtime.md](h3-modal-runtime.md) | The Modal runtime |
| [h3-modal-gpu-selection.md](h3-modal-gpu-selection.md) | Single-GPU Modal selection |
| [h3-model-storage.md](h3-model-storage.md) | Model storage and staging |
| [h3-production-inputs.md](h3-production-inputs.md) | Production inputs |
| [h3-production-media.md](h3-production-media.md) | Production media handling |
| [h3-checkpoint-handoff.md](h3-checkpoint-handoff.md) | Checkpoint handoff between stages |
| [h3-zoom-integration.md](h3-zoom-integration.md) | Zoom as an input source |

Working notes, dated failure reports, and superseded plans are not published; see [`../research-log/README.md`](../research-log/README.md).
