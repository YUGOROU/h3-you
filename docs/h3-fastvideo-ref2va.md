# Native Ref2VA + original64, isolated FastVideo implementation

## Scope and status

**Realtime+identity follow-up:** `h3-fast-ref2va-smd.md`. The native50-forward mode
below is diagnostic only, not the realtime solution. New explicit `--fast-ref-smd`
mode uses a Ref2VA-trained four-forward candidate and requires separate GPU gates.

USER rejected the personal T2VA result as an unrelated person and requested actual
Ref2VA inference. This implements a **separate native Ref2VA path**, not a rename
of the failed T2VA experiment. Original64/Baseline/715cache and previous runtime
artifacts are untouched. No new GPU allocation/submission/upload in this step.

Implemented and CPU-tested; **GPU execution, identity efficacy, visual quality and
speed are unverified**. This is initially a one-request correctness pilot, not the
four-segment realtime producer. Do not carry forward the T2VA 1.53x speed claim.

## Actual inference contract

- FastVideo `ba371e33ed080e51f1884ae166baf7c4447a66fe`, same pinned CUDA image and
  dependencies as `h3-realtime-first-experiment.md` (Torch2.12.0/kernel0.3.5/quack0.5.3).
- Explicit `ComponentConfig.override_pipeline_cls_name = MiniMaxH3Ref2VAModularPipeline`.
  That class sets Ref2VA mode and maps transformer → **`transformer_ref`**.
- Original `MiniMaxAI/MiniMax-H3` @`42ed227ee7df40d41602854ae760620d6eb651fe`,
  native **14-shard transformer_ref**. Not the separate curated13-shard Diffusers
  asset repository. Dedicated Ref2VA-only snapshot excludes the T2VA partition.
- Request uses `InputConfig(references=[MiniMaxH3Reference(source=..., media_type='image')])`.
  **Not** `image_path`, `pil_image`, or previous generated frames (those are FL2VA
  keyframe fields and explicitly rejected by this pinned Ref2VA input stage).
- Original64 raw adapter SHA
  `43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151`,
  rank16/alpha16, strength1,208pairs/416tensors. No rank80 derived T2VA artifact.
- Both worker ranks must attest actual Ref2VA mode, transformer_ref mapping,
 208rank16 pairs applied, total208adapted layers, zero unconsumed adapter tensors.
  The pinned loader is overlaid **only in a new private runtime copy**; original
  source SHA `9db2b62897bc5f14e8cf117c5467875c41498856a6d1893c6e15a41ecab0652c`,
  Ref2VA overlay SHA `2f282b74cb5809ba01f23ab3e9fda8af630d320cee958a09277e4646aaf223fd`.
- 256x448,124frames,24fps, seed314159; original native schedule51sigma-grid points
  (expected50forwards). Guidance1. Checkpoint video/audio scheduler settings are
  loaded by FastVideo, not rewritten. Fixed prompt explicitly names reference1.
- Dense FLASH_ATTN, SP2/TP1 on exactly2H200, compile off, parallel VAE gather.
  **No** T2VA FastH3 four-step/VSA adapter, Turbo, FP8 or speculative stacking.
  Native50-forward inference may be much slower. Realtime acceleration must use
  a separately verified Ref2VA-compatible adapter/checkpoint, not the failed T2VA mix.
- Image preparation and ordered conditioning use the pinned Ref2VA source's
  released2048-short-edge aspect-preserving preparation (resolver returns H,W).
- Output is **visual-only**: generated joint audio is discarded. No supplied WAV,
  Irodori audio conditioning, original-PCM mux, streaming, OBS or Discord claim.

## Files / usage

- `research/prepare_h3_ref2va.py`: metadata only, no weights/uploads/allocations.
- `research/h3_personal/ref2va_runtime.py`: request/config, hashes, actual-load guards.
- `research/run_h3_ref2va.py`: defaults to full offline preflight; explicit `--run`
  only executes on an already provisioned matching CUDA runtime. No provider API,
  automatic submission, publication, retries, installs or fallback.

Metadata manifest prepared privately at
`$H3_PRIVATE_ROOT/ref2va-fastvideo-v1/model-manifest.json`:
**63files/144051199021bytes**, metadata verified at the original public revision.
No large model weights were downloaded to the Mac. The manifest is a trusted local
receipt from the pinned Hub metadata; do not substitute an unreviewed manifest.

On a provisioned runtime with the exact listed63files already available:

```sh
python -m research.run_h3_ref2va \
  --model /private/native-ref2va-snapshot \
  --model-manifest /private/model-manifest.json \
  --identity /private/original64/adapter_model.safetensors \
  --identity-config /private/original64/adapter_config.json \
  --reference /private/original-own-reference.png \
  --reference-sha256 6789db42c765d2611a640bc9b47934b4912584e45c800d2700b1fddf9902faeb \
  --runtime-archive /private/FastVideo.tar.gz \
  --output /private/new-ref2va-review
```

Default checks hashes of all listed files and exact local inventory,14shards/index,
original adapter/config, reference hash and oriented geometry. It performs no GPU
inference. Do not use the screenshot of the unrelated generated person as reference.
An explicitly chosen new own reference requires its own independently checked hash.

Use the prior successful deployment's source archive (unchanged public code-only
artifact), SHA `6ef5b1ebb373cc4d09e8e5de1851d8acf6c23c83cbe5b60c2e86eafaeec3f1c7`.
A freshly recompressed archive can have a different byte hash; do not bypass it.
Add `--run` **only after** normal live price/idleness/privacy/permissions/region
acceptance/pins/budget checks and an externally enforced finite Job timeout. This
runner is not a new authorized paid launcher. Current unreserved authority4.20USD.

Execution creates a new private output directory, extracts verified code, patches
only the isolated copy, reexecs with that source and inherited worker PYTHONPATH,
checks dependencies/H200s and the shared actual FFmpeg encoder probe, then requires
both load receipts before generation. Saves `review.mp4` and `profile.json` locally,
validates packets/full decode; no autoplay or upload. Shut down generator on error.

## Verification / remaining gates

- Dedicated CPU tests8/8, including actual pinned API dataclasses and the reference
  request type (archive supplied explicitly), model-hash/index fixture rejection,
  wrong T2VA route, unmatched keys/ranks, no-submit/no-upload defaults.
- Full stdlib suite577OK/85skipped. Actual cached original64 full SHA and
  raw BF16 header208pairs also revalidated without Torch/model execution. Source-overlay AST compilation verified.
- Reuse search returned no matching published asset. Reused existing pinned
  FastVideo Ref2VA implementation, identity validator and FFmpeg probe/encoder;
  no additional dependencies installed.
- Model fixtures and source/API tests are **not GPU model/LoRA parity or identity
  evidence**. Next: bounded isolated GPU smoke, human identity review, then a
  verified compatible Ref2VA acceleration and actual streaming/audio path.
