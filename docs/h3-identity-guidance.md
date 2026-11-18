# Pinned Aviad guidance-target correction (experimental subset)

**Gate C remains disabled.** This implements a source-level, CPU-tested subset, not a demonstrated likeness fix or authorization to train. No weights, downloads, jobs, old launchers/configurations, or a local status note were changed.

## What is implemented

The installer accepts the exact Aviad source snapshot `7a1563a267a17da898df3396acbd6a9d24fcb142`, verifies SHA256 for **every Python file under `src/`** and `scripts/process_dataset.py`, checks unique replacement contexts, and parses all resulting Python. Drift, extra source files, missing files, source-file symlinks, and already-patched sources fail closed. It publishes a **new source-only tree**, never patches the input tree. No imaginary model/cache interface or monkeypatch is used.

Actual changes generated in that new tree:

- `config.py`: optional `training_strategy.identity_guidance_scale` (default `null`, disabled) and `identity_guidance_sigma_min` (default `0.15`). Enabled strategies add the real dataset source `identity_uncond`.
- `scripts/process_dataset.py` / `preprocessing/builder.py`: explicit `--identity-guidance-uncond` flag. With `--references`, the actual `TextPass` rebuilds the image presentation once and passes **the same reference descriptors/pixels** into `H3Encoders.encode_ref2va_prompt` for the caption and a single-space text probe. Existing conditional caches are re-encoded in this explicit mode; incompatible samples raise rather than get silently skipped. There is no new encoder/model loader.
- `datasets.py`: actual `PrecomputedDataset.__getitem__` loads per-sample probe embeddings/tags and verifies provenance and reference vision-token presence. Batch-1 `bucket_collate` passes these through; larger batches are rejected.
- `training_strategies/flexible.py`: actual `prepare` assembles/noises the conditional batch **once**, then builds the probe's text-dependent indices and row timesteps. `dataclasses.replace(PackedBatch, ...)` retains the exact video/audio tensors, including augmented image rows, clean target audio, noise, masks, and target geometry. Actual `compute_loss` consumes the corrected video target.
- `trainer.py`: actual training **and held-out loss validation** call the helper around `self._forward`. The detached probe runs **before** the grad-enabled conditional forward. The model, active LoRA, train/eval mode, and dtype policy are unchanged. No adapter disabling, frozen teacher, or training adapter is introduced. Startup rejects unsupported configurations before model loading.

The installed helper is `src/h3_trainer/identity_guidance.py`, copied from `research/h3_personal/identity_guidance.py`. The installer does not copy other scripts, environment installers, configs, dependencies, weights, artifacts, or caches. Use the project's separately managed pinned trainer environment for imports; this is not a standalone trainer distribution.

## Exact math and invariants

Following pinned Musubi `f8a1b03794a49239a3539015075f5123d6c07d66`, `minimax_h3_train_network.py:1279–1291,1369–1447` (see [audit](h3-identity-loss-audit.md)):

```text
raw_video = clean_video - noise_video                 # data-ward velocity
if SigmaPair.u >= sigma_min:                          # PRE-SHIFT draw, inclusive
    U = active_LoRA_model(text_probe, same_video_audio).video.detach()
    target_video = U + scale * (raw_video - U)         # fp32, generated rows only
else:
    target_video = raw_video                          # no extra forward
loss = MSE(conditional_video[video_loss_mask].float(), target_video)
```

Targets are detached, shapes must match exactly (no broadcasting), and reference prediction rows are excluded from loss. Audio remains clean at `t=1`, `audio.is_generated=false`, `audio_loss=0`, `audio_weight=0`. No audio target correction or audio-generation loss is added. The shared attention LoRA can still change audio-conditioned behavior; audio preservation is not proved by this invariant.

The single-space probe is text-unconditional, **not reference-unconditional**. A global text-only Musubi cache is incompatible. Probe text length can differ: Diffusers Ref2VA moves absolute indices **and the shared rotary time origin** with text length. The spatial grids, row order, masks, and relative video/audio clock must remain unchanged; runtime checks enforce this, allowing only the common temporal translation. No caption padding, embedding zeroing, reference dropping, or second noise draw is used.

`scale=4`, `sigma_min=0.15` are upstream heuristics, not a calibrated personal-training recipe. Scale must be finite and >=1; threshold finite in [0,1]. Scale 1 is algebraically raw flow, though an enabled high-sigma probe still incurs a forward.

## Cache contract (fail closed)

Under the actual preprocessing root:

```text
conditions/<id>.safetensors           # conditional prompt_embeds + text_token_tags
reference_latents/<id>.safetensors    # existing reference rows + geometry
identity_uncond/<id>.safetensors      # paired probe prompt_embeds + text_token_tags
```

Probe safetensors metadata must exactly contain:

- `schema=h3-ref2va-text-uncond-v1`
- `aviad_revision=7a1563a267a17da898df3396acbd6a9d24fcb142`
- `sample_id=<id>` and `probe_prompt=" "`
- `conditional_sha256` and `reference_sha256`: hashes of those exact cache files.

Missing probe, missing real audio, stale conditional/reference hashes, incompatible shapes/tags, reference-free vision blocks, reference audio, or non-image reference geometry raise. There is no fallback even below the sigma threshold. Rebuilding a conditional/reference cache invalidates the paired probe until rebuilt. Metadata binds files, not an independently authenticated semantic claim; do not hand-label arbitrary embeddings as compatible.

## Supported subset and remaining blockers

Only **single-process, unquantized `ddp`, micro-batch 1, no/bf16 precision, zero-dropout LoRA, Ref2VA, one always-on image reference per sample, generated video and clean real target audio** is accepted. Different samples may use different photos; this restriction excludes multiple simultaneous reference rows, not a multi-photo dataset. DeepSpeed/ZeRO, distributed DDP, model-parallel, quantized models, multi-reference/video/audio-reference caches, reference dropout, and larger batches are rejected. This avoids pretending CPU tensor tests prove extra-forward collective/offload safety.

Remaining acceptance/interface blockers:

1. **No real H3/PEFT forward/backward was run.** A differentiable stand-in verifies forward order, active trainable delta, detached probe, and loss integration. GPU memory, actual gradient-checkpointing/PEFT execution and clean-audio inference parity remain unverified. No identity, detail, lip-sync, or preservation result exists.
2. **Existing cache provenance is insufficient to authenticate source pixels.** Pinned Aviad's media index does not bind original image bytes/encoder-weight revisions to cached reference latents. The new producer uses identical pixels for both text branches and binds the resulting files, but cannot prove a supplied old latent cache came from those pixels. A separately reviewed, authorized full media/text recache with recorded source/model provenance is needed before acceptance. The code does not read/download model weights to establish this.
3. **No runtime dependency/weight revision attestation.** Source hashes pin Aviad, not installed Diffusers/Transformers/PEFT or base weights. These must be independently checked before any pilot.
4. The pinned trainer's reference-inclusive sequence/memory budgeting and unrelated `_skipped_total` initialization issue are unchanged. Larger-layout/skip-path execution is not validated by this subset. Held-out loss now measures the guided objective, so it is not directly comparable to old raw-flow loss curves.
The composed [component-cache installer](h3-component-cache.md) now fixes reference/probe-inclusive sequence budgeting, initializes the missing skip counter, and fails on v2 token-limit sample loss. The guidance-only installer intentionally remains the smaller source patch.

5. More general modalities/batching/backends need explicit interface work and tests, not relaxing the guards. No automatic migration/resume or gate-C launcher wiring is provided.

`C_GATE_ENABLED = False` records this status; it is **not** an authorization token or a replacement for external launch gates. The default strategy remains disabled. Old launchers are untouched. Do not use the experimental settings to bypass gate C.

## Safe source verification and tests

No torch is required for installer/static tests:

```bash
python3 research/install_identity_guidance.py /tmp/h3-identity-audit/aviad
# Optional: generate source only, in a destination that does not exist:
python3 research/install_identity_guidance.py /tmp/h3-identity-audit/aviad \
  --output /tmp/h3-guidance-source-review
python3 -m unittest discover -s tests -v
```

Torch tests skip when torch is absent. Keep ML dependencies in a separate environment. Local source-plan/integration tests use `H3_AVIAD_AUDIT_SOURCE` (default the audit snapshot); the snapshot-dependent cases skip if unavailable, while stdlib hash/context/installer tests still run.

Verified locally:

- Root stdlib suite: **102 tests, 5 skipped**, passing.
- Separate existing `/tmp/h3-media-env`: **12 guidance tests**, passing, using torch `2.14.1`, safetensors `0.8.0`, and Diffusers whose installed provenance reports commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.
- The CPU integration installs the exact patch into a temporary tree, exercises actual `PrecomputedDataset`, `bucket_collate`, `FlexibleStrategy.prepare/compute_loss`, and real pinned Diffusers Ref2VA packers on synthetic tensors. It checks differing text lengths, identical reused inputs, spatial/relative-time parity, audio cleanliness, masks, threshold, gradients, and missing/stale/reference-free cache rejection.
- The encoder pairing test uses a stand-in encoder and checks caption vs single-space prompts share the same reference descriptors. It does **not** execute Qwen3-VL.

The guidance work itself performed no weights/model inference/training/media recache/network/cloud job. Source/tensor success does not open production gate C. Separately, quality-selection CPU processing is recorded in h3-quality-cpu.md.

`research/configs/ref2va_identity_smoke.example.yaml` is an unlaunched3-update schema-checked example with guidance4, pre-shift threshold0.15, provisionalLR1e-5, rank/alpha16, clean audio, zero dropout and single-processDDP. Validation used empty dummy directories, not real model/cache artifacts. These are starting parameters, NOT measured production throughput or likeness evidence. Don't blindly run the old launcher with this file. Actual encoder/catalog pipeline, provenance, runtime weights/environment, permissions/territory and GPU cost must be established first; minimal authorized smoke checks real H3/PEFT interfaces before production.
