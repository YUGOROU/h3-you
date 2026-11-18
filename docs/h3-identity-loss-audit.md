# H3 identity / distilled-training loss audit

## Subsequent implementation

This document preserves the historical source audit. The project has since added an exact-pinned, CPU-tested [guidance-target path](h3-identity-guidance.md) and [shared component loader](h3-component-cache.md), without a training adapter or trainer/task replacement. Real encoder/provenance/GPU acceptance remains open. USER now requests minimal1–3-update implementation smoke followed directly by production, not repeated pilots; early production checkpoint evaluation supplies quality evidence.

## Scope and conclusion

Source-only investigation of the **actual pinned AviadDahan training objective** and public Musubi/Fizgig mitigations. Public source archives and one small model README were downloaded into `/tmp/h3-identity-audit/`; no model/adapter weights, HF uploads, jobs, training, or inference were performed. No fixes are implemented by this audit.

**Conservative gate: do not extend/resume the existing plain-flow identity run or switch trainers automatically.** The pinned run optimizes raw velocity against a guidance-distilled Ref2VA base without a training-time de-distillation adapter, guided-target reconstruction, or frozen-teacher target. Public implementations treat that mismatch as a training risk. This is a credible explanation for quality degradation, **not a demonstrated cause** of this user's poor likeness/coarse output, and not evidence that an adapter will repair it.

The existing checkpoint's strict loading and completed optimization establish neither identity quality nor preservation of the base's distilled prediction distribution. User rejection of the old- and new-reference results remains authoritative; the canceled comparison's remaining cases are not evidence.

## Immutable evidence pins

Line numbers below refer to these snapshots, not current branches. GitHub source links use `blob/<commit>/<path>#L…`.

| Component | Repository / revision |
| --- | --- |
| AviadDahan trainer used by this project | [AviadDahan/MiniMax-H3-Trainer](https://github.com/AviadDahan/MiniMax-H3-Trainer/tree/7a1563a267a17da898df3396acbd6a9d24fcb142), `7a1563a267a17da898df3396acbd6a9d24fcb142` |
| Musubi public source inspected | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner/tree/f8a1b03794a49239a3539015075f5123d6c07d66), `f8a1b03794a49239a3539015075f5123d6c07d66` |
| Fizgig public source inspected | [shootthesound/Fizgig](https://github.com/shootthesound/Fizgig/tree/8689f436587e8c26fe98e2103cd913337c2c88e4), `8689f436587e8c26fe98e2103cd913337c2c88e4` |
| Circlestone README only, independently retrieved | [circlestone-labs/MiniMax-H3-Image-Training-Adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter/blob/7c6c04f75ba597232d4ce2bb4a14709948ec723a/README.md), `7c6c04f75ba597232d4ce2bb4a14709948ec723a` |
| Existing project dependency pins (not re-downloaded/evaluated here) | H3 `42ed227ee7df40d41602854ae760620d6eb651fe`; Diffusers `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` |

Musubi/Fizgig pins are inspection snapshots, **not dependencies used in the completed training run**. Anonymous access to `https://huggingface.co/api/models/ostris/minimax_h3_training_adapter` returned HTTP 404 during inspection. Its availability, weight revisions, metadata, and contents were therefore not independently verified. A 404 does not establish whether the repository is deleted, private, renamed, or temporarily unavailable.

## 1. What the pinned Aviad trainer actually optimizes

### Correct raw-flow conventions, but no guided-space target

[Source: `src/h3_trainer/flow_matching.py`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/flow_matching.py#L32-L125):

```python
# shift_sigma / timestep_from_sigma / add_noise / velocity_target
return shift * u / (1.0 + (shift - 1.0) * u)
return 1.0 - sigma
return (1.0 - sigma) * clean + sigma * noise
return clean - noise

# flow_matching_loss
 target = velocity_target(clean, noise)
 pred = prediction.float()
 target = target.float()
 # row_mask selects generated prediction rows, not conditioning rows
 return torch.nn.functional.mse_loss(pred, target) * weight
```

The excerpt combines separate functions; indentation is illustrative. Video/audio sigmas are derived from a shared `u` (`SigmaPair.from_u`, same file). The signs and reversed time convention are intentional: H3 predicts **data-ward** velocity `x0 - eps`, with `t=1-sigma`. Do not "fix" this to the common noise-ward sign or feed sigma directly to H3.

[Source: `training_strategies/flexible.py:79–112,201–241`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/training_strategies/flexible.py#L79-L112):

```python
sigma_video = sigmas.video if self.config.video.is_generated else 0.0
sigma_audio = sigmas.audio if self.config.audio.is_generated else 0.0
# assemble receives timestep_from_sigma for each modality

# compute_loss (201–241)
video_loss = flow_matching_loss(prediction_video, packed.clean_video,
                               packed.noise_video, row_mask=packed.video_loss_mask)
# Audio loss exists only when audio.is_generated is true.
# Otherwise weight=0.0, audio_loss=zero.
total = video_loss + weight * audio_loss
```

`trainer.py:436–447` prepares the packed batch, performs the model forward, calls `strategy.compute_loss`, and backpropagates that total. There is no unconditional-probe/teacher forward on this path. The trainer itself acknowledges released checkpoints are guidance-distilled in [`config.py:400–408`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/config.py#L400-L408), but that acknowledgement concerns inference/validation CFG, not a training correction.

### Exact relevance to this project's run

Local `research/configs/ref2va_audio_lock_train.yaml` selects:

- `model.variant: ref2va`, rank/alpha 16, attention targets `[to_q,to_k,to_v,to_out.0]`;
- generated video; image-reference probability 1.0;
- `audio.is_generated: false`: target driving audio clean at `t=1`, no audio-generation loss;
- 120 steps, LR `0.0001`, video/audio shifts 12/3, no held-out video samples.

The strategy therefore trains **video MSE against raw `clean_video-noise_video`**, conditioned on reference image and clean target audio. Removing audio loss does not correct the video's guided-space mismatch; shared trainable attention can still alter audio-conditioned behavior. No face-embedding, face-region weighting, perceptual identity, or dedicated identity objective appears in this loss.

[`model_loader.py:30–72`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/model_loader.py#L30-L72) explicitly maps `fl2va` to `transformer` and `ref2va` to `transformer_ref`. These are different task variants, not interchangeable labels.

[`flexible.py:139–151`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/training_strategies/flexible.py#L139-L151) also documents a separate reference-dropout gap: dropped latent rows retain the reference-bearing cached text/vision presentation. The project's probability 1 avoids this gap; reducing it is **not** a safe identity mitigation without matched reference-free caches.

## 2. Musubi: three distinct mitigation mechanisms

[Recipe documentation, `docs/minimax_h3.md:109–204`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/minimax_h3.md#L109-L204) identifies CFG-distilled weights and warns that plain-flow training washes out video/prompt adherence and can rapidly break image structure. Those quality/timing statements are **upstream reports**, not measurements of this project.

### A. Training-only de-distillation adapter

`--base_weights adapter.safetensors` merges a frozen adapter into the base before training; the learned personal LoRA is then used on the original base **without** that training adapter. This is not a Turbo inference LoRA and does not change the mathematical raw-flow loss itself.

- [`minimax_h3_train_network.py:850–860`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/minimax_h3_train_network.py#L850-L860) converts adapter keys and delegates merging; a separate branch handles already-merged load-time weights.
- [`networks/lora_minimax_h3.py:15–38`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/networks/lora_minimax_h3.py#L15-L38) accepts native keys and upstream `diffusion_model.` / `transformer.` `lora_A/lora_B` format, filling absent alpha as rank. Default native trainable targets include fused attention **and MLP**: `blocks.N.(attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2)`.
- Documentation requires a mergeable BF16 source (optional runtime ConvRot INT8 afterward), not an already pre-quantized INT8 file. Training-time samples include the merged correction and are not the final deployment combination.

This capability exists upstream; **the pinned Aviad trainer has not been modified to provide it**.

### B. Guidance-preserving target reconstruction

[`minimax_h3_train_network.py:1279–1291,1369–1447`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/minimax_h3_train_network.py#L1369-L1447) starts from raw velocity but reconstructs a guided target:

```python
video_target = latents - noise
# applies above the configured base-sigma minimum
with torch.no_grad(), accelerator.autocast():
    uncond = transformer(...)  # text swapped; visual/audio conditions retained
uncond_video = uncond.video.detach().float()
video_gap = video_target.float() - uncond_video
video_target = uncond_video + float(args.h3_guidance_loss_scale) * video_gap
```

The LoRA **remains active** during this no-grad probe. This is text-unconditional, not removal of the visual/audio conditions. Audio has a parallel target with separately configurable scale. The documented example is text cache `--uncond_output`, then `--h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15 --h3_guidance_loss_uncond_cache …`; these are upstream heuristics, not a verified H3-personal recipe. There is an extra forward above the gate, not free mitigation.

### C. Frozen-teacher matching

[`minimax_h3_train_network.py:1470–1520`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/minimax_h3_train_network.py#L1470-L1520) disables the learned LoRA for a teacher forward. Teacher-only privileged conditions can be crop endpoints (`first,last`), the clip itself (`ref`), or other subject images (`subject_ref`). The student is **T2VA**, not the current reference-conditioned audio-locked student.

The teacher supplies its prediction instead of raw velocity. Conditioned teaching is restricted to a sigma band; outside it, the teacher uses the student's unconditioned layout as a base-preservation anchor. Comments explicitly warn unrestricted high-noise teaching can overwrite composition with dataset-mean content. This is an identity-learning design with extra compute/cache and changed deployment task, not a drop-in loss flag for this run.

[`minimax_h3_train_network.py:835–848`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/minimax_h3_train_network.py#L835-L848) warns against combining training adapters with guided-target correction; with teacher matching, the merged/de-distilled base becomes the teacher. Guidance loss and teacher matching are mutually exclusive. Do not stack all three "for safety."

## 3. Fizgig: frozen adapter lifecycle, not a hidden corrected raw target

Actual [`families/train.py:960–975`](https://github.com/shootthesound/Fizgig/blob/8689f436587e8c26fe98e2103cd913337c2c88e4/src/fizgig/families/train.py#L960-L975):

```python
n = net.add_file(training_adapter, ADAPTER, training_adapter_strength)
if n == 0:
    raise RuntimeError(...)
# logged: frozen, off in previews, not saved into the LoRA
```

The preview path calls `net.set_enabled(ADAPTER, False)` at line 404 and restores it at line 480. Frozen correction and personal trainable LoRA are separate components. A nonzero module match is useful but weaker than proving complete key coverage or correct base semantics.

[`minimax/common.py:978–1014`](https://github.com/shootthesound/Fizgig/blob/8689f436587e8c26fe98e2103cd913337c2c88e4/src/fizgig/minimax/common.py#L978-L1014) still uses:

```python
noised = (1.0 - s) * x0 + s * noise
t = (1.0 - sigma).to(device)
loss = F.mse_loss(pred.float(), (x0 - noise).to(pred.dtype).float())
```

With real audio latents, this path noises audio on its own schedule and adds audio MSE. **It is not equivalent to this project's clean-driving-audio training contract.**

Fizgig also implements reference distillation in [`minimax/common.py:1270–1338`](https://github.com/shootthesound/Fizgig/blob/8689f436587e8c26fe98e2103cd913337c2c88e4/src/fizgig/minimax/common.py#L1270-L1338): `w*MSE(student,teacher)+(1-w)*MSE(student,x0-noise)`, with shared noised input/audio silence and reference-bearing teacher versus text-only student. [`minimax/driver.py:32–35,406–426`](https://github.com/shootthesound/Fizgig/blob/8689f436587e8c26fe98e2103cd913337c2c88e4/src/fizgig/minimax/driver.py#L406-L426) shows the teacher turns off **only trainable** LoRA; frozen training/context adapters stay on. This differs from a pristine guided-base teacher. Reference/identity distillation must not be conflated with preserving the released model's CFG distillation.

## 4. Ref2VA vs FL2VA compatibility: structural load is not semantic proof

| Asset | Musubi's pinned documentation | Fizgig's pinned source | Conservative interpretation here |
| --- | --- | --- | --- |
| Ostris standard adapter | FL2VA **v2**, rank 32 | Standard FL2VA **v1** filename | Different versions are referenced; do not treat as one pinned asset. Not established for Ref2VA. |
| Ostris `minimax_h3_ref2va_training_adapter_v1.safetensors` | Ref2VA, rank 16 | Auto-selected for Ref2VA or reference-distillation runs | Variant-aligned candidate for the current base, but availability/content/revision unverified (anonymous 404). |
| Circlestone image adapter | FL2VA, rank 64, image-first; limited video reports | Advertises **one file for both FL2VA and Ref2VA**, default, stronger still-image likeness | Cross-base assertion conflicts with the narrower Musubi description; no Ref2VA quality/compatibility proof obtained. |

Evidence: [Musubi adapter table `docs/minimax_h3.md:171–183`](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/minimax_h3.md#L171-L183); [Fizgig `families/minimax.py:241–265`](https://github.com/shootthesound/Fizgig/blob/8689f436587e8c26fe98e2103cd913337c2c88e4/src/fizgig/families/minimax.py#L241-L265). Musubi explicitly says all three load/train but **output quality has not been evaluated there**. Fizgig's sharper-eyes/likeness language is upstream product/source text, not independently reproduced evidence.

The pinned Circlestone README says the correction both undistills and biases toward images, deliberately absorbing image-only video-knowledge degradation. It instructs training with correction, inference without it, and **DO NOT combine with CFG-augmented/guidance-preservation training**. It calls video/mixed performance lightly tested. It does not independently certify Ref2VA suitability.

Key-format trap: Musubi calls fused original-style `diffusion_model.blocks.N.attn.qkv_proj.lora_A/B` a "Diffusers" format. Aviad's actual Diffusers/PEFT modules instead use split `transformer_blocks.N.attn.to_q/to_k/to_v/to_out.0`. Its [`lora.py`](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/lora.py#L1-L19) explains exact QKV export re-fusion (rank-r separate updates become rank-3r fused update). **Naming alone does not establish an existing importer into Aviad.** Frozen-adapter conversion/merge would require checked prefixes, QKV splits, shapes, rank/alpha scaling, full module coverage (including refiners/MLP if present), and numerical delta equivalence. These are unimplemented/unverified for third-party training corrections here.

## 5. Blockers and recommended acceptance gate

1. **Do not diagnose identity failure from loss alone.** No matched LoRA-off base comparison, independent held-out identity evaluation, face-detail/lip-sync assessment, or mitigation A/B exists here. Shared backgrounds/clothing, identical reference presentations, correlated video clips, limited exposure (120/489 ≈ 0.245 epoch), and preprocessing/detail limits are alternative or interacting hypotheses. More steps may amplify drift rather than improve faces.
2. **Keep the intended task explicit:** Ref2VA plus frozen clean target speech is the current target. Musubi T2VA teacher recipes and Fizgig's noisy real-audio/image distillation paths change that contract. A trainer migration must separately establish full-video geometry, ordered reference/text tags, clean-audio `t=1`, masks, and inference parity. No cache interchangeability is established.
3. **Select one mitigation only after approval.** For an unchanged Ref2VA route, evaluate a revision-pinned, explicitly Ref2VA training-only correction if it becomes verifiably available; alternatively investigate a guided-target implementation preserving clean audio. Neither is already integrated. Do not substitute FL2VA/Circlestone on filename or shape match alone. No recommendation to train/use Turbo as the correction.
4. **Before any weights/compute are authorized:** resolve adapter availability, variant, license/territory, provenance and immutable revision; review conversion/merge coverage and frozen-vs-trainable behavior. Static/CPU tests should prove target math, detached probe/teacher behavior, masks, clean-audio contract, alpha scaling, separate export, and removal of correction at deployment. This audit grants no download/job authorization.
5. **Only newly authorized bounded execution with actual generated outputs may establish quality:** minimum smoke checks runtime, not likeness; evaluate early production checkpoints without repeated pilot campaigns. same base/portrait/audio/prompt/seed/schedule, separate held-out capture/reference, LoRA-off control, early-checkpoint evaluation, identity and image-detail review plus lip-sync/audio preservation. Compare one mitigation at a time; preserve pristine base and existing checkpoints. Stop on washed-out/detail collapse or failure to improve likeness; never infer success from decreasing MSE, successful loading, or adapter UI claims.
6. Any future cloud work still needs explicit user approval, license/execution-location checks, finite timeout, no duplicate jobs, and cumulative budget <=$150 / rate <=$10/hr. No automatic retraining or restart of canceled comparisons.

**Status:** findings only. No trainer, loss, cache, adapter loader, checkpoint, inference path, or research configuration was changed. The immediate safe action is to retain the failed-quality evidence and keep further compute gated, not announce an implemented fix.
