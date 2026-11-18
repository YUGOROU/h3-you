# H3 aggressive acceleration: isolated experimental Job

## State

Implemented and submitted as an **opt-in separate Job**, not an in-place control modification.
Corrected experimental Job `6ad0c85fc66288145225170a` COMPLETED;
BF16 control `6ad0a992c66288145224fe45` COMPLETED and generated video/PCM verified,
but USER rejected visual ghosting/background instability. See causal exclusions.
Previous FP8 Job `6ad0a99ac66288145224fe4e` ended ERROR in the pre-denoise probe. Prior control
`6ad078215ccb47ba0f7a5e6a` ended ERROR.
That failure was rank1 remote Turbo read EIO followed by a20-minute peer collective timeout.
Both prepared branches now stage Turbo once to a SHA-verified local file before model/torchrun,
retry only EIO up to3 times, use explicit NCCL devices and a300-second collective timeout.
Fresh private runtime:

- Dataset `YUGOROU/h3-fp8-streaming-runtime-20261015-v5`
- Revision `0b815756f697b2fabc0b7441f58d59bbfd1e2026`; deployment revision `36229fc9fc734ec0b3d0746b800c9fd5ee62cce3`
- Private local evidence: `$H3_PRIVATE_ROOT/gpu-hf-fp8-streaming-experiment-v5/`
- Full tests: **528 OK / 79 skipped**; pinned TorchAO scale regression/wiring **5/5 pass**; dedicated pinned VAE protocol tests **3/3 pass**.
- Both real deployments passed full local/remote pin/privacy/rate gates before allocation.
  Each has its own private immutable source/runtime/environment, finite4500s timeout and12.500025USD reserve.
- USER reported actual HF107.80+Modal35.59=143.39 and approved170USD cumulative/two parallel Jobs.
  Past terminal estimates archived, NOT added again. Keep1USD ancillary margin; current reconciled bound161.37 (control6.56+failedFP84.08+completedFP86.34+baseline143.39+margin1).
  Defaults remain150USD/one-allocation rates; `execution_budget.py` gates the narrow pair exception.
  At most one active Job per authorized scope; unrelated active Jobs, unknown allocations, missing
  ledger approval or changed authority fail closed. Terminal corrective retries still require diagnosed
  fixes/fresh immutable deployments and reconciled reservations within current cumulative authority.
- FP8/PEFT/CP completed40forwards; actual VAE pixel parity and visual acceptance remain unverified.

### Completed combined experiment results

Output production repo @`0b9d1708297768aa72b3d0b147bd25525fea97e8`, prefix
`identity/h3-4d0569831763490d9c827fdb292641e9/acceleration-experiment/`.
Recovered MP4 and both profiles; remote bytes/hash,683frames768x1344, full FFmpeg decode,
and original24bit PCM equality verified locally. Video SHA
`bc3987c76c2a77d84e6b7abd47a060d76dcf06789b048bfb3a9ea0681fcb7729`.

| Metric | BF16 control | Combined FP8/short/incremental |
|---|---:|---:|
| Denoise seconds |565.161|521.666|
| Decode/export phase seconds |301.316|311.397|
| Sum seconds |866.477|833.063|
| Denoise peak allocated GB/device (decimal) |82.355|69.818|
| Decode peak allocated GB (decimal) |35.913|12.135|

About4% less summed denoise+decode time, not realtime. This changes three factors,
so it is NOT an isolated FP8 speed/quality measurement.300selected Linear modules converted;
1040named adapter tensors unchanged; finite zero/nonzero and `aten::_scaled_mm` observed.
Incremental decoder:40clips, first RGB7.198s/packet22.636s **after denoise**, final MP4 only;
no live transport, physical-flush, overlap or causal claim. No same-latent GPU pixel parity.
First-clip extraction (frames0,1,2,3,4,5,8,12,16) still shows early doubled outlines;
short windows/FP8 did not establish an artifact fix. Full human quality review remains pending.
Job runtime2281.097s ->6.34USD rounded-up time-rate bound, not invoice. No additional Job launched.

Public reuse search found no published matching asset. Reuse existing native
continuation, named LoRA composition, strict per-rank reload, CP and generated-only
upload. Use official TorchAO API rather than inventing a quantizer or migrating the
whole pipeline to another model/framework.

## Corrective FP8 scale/probe fix

The failed v4 gate combined `not finite` and `scaled-mm not observed` into one error,
so its log does not independently identify which subcondition failed. Pinned TorchAO0.13
v2 source reproduces zero input -> zero scale -> NaN with the old all-zero probe.
Exact source function bodies (only registry decorators stripped) were tested on CPU:
old zero probe fails, positive floor keeps zero finite, normal nonzero scales/values unchanged.
This is source-math evidence, not a claim of completed CUDA/CP correctness.

`fp8_recipe()` now sets `activation_value_lb=1e-12` and explicit `KernelPreference.TORCH`.
The real selected model Linear must pass BOTH zero/nonzero finite probes AND actual
`_scaled_mm` profiler evidence; checks are separated and diagnostics retained, not removed.
`prepare_h3_fp8_runtime --execute` now tests tiny256x256 real FP8 Linear on each H200
with both fast-accum modes before expensive model inventory/conditioning/reload.
It does not modify model weights, adapters, package versions, BF16 control, or enable compile.
A passing tiny kernel still would not prove PEFT/CP integration, VAE pixel parity or quality.

Terminal reconciliation uses started-to-finished runtime (2358.402s/6.56USD and1465.774s/4.08USD,
rounded up cents), not the FP8 Job's50-minute scheduling queue. These are not invoices.
Original12.500025 reservations and complete timestamps remain in private ledger receipts.

## Experimental controls

All modes preserve original64, named Turbo alpha8, source assets, original WAV,
corrected reference policy, seed314159, shift12 and9grid/8forwards per window.
The original HF model/adapter repositories are never overwritten. Quantization is
in memory in a fresh Job; no quantized model or modified adapter is uploaded.
The original producer3 source files remain byte-identical for frozen715cache.

| Preset | Base Linear quantization | Window frames | Decoder |
| --- | --- | --- | --- |
| bf16-control | none | 277 | existing full decode |
| fp8-ffn | block FFN only | 277 | existing |
| fp8-blocks | block attention+FFN | 277 | existing |
| short-native | none | 158 | existing |
| incremental-native | none | 277 | incremental native chunks |
| fp8-short-incremental | block attention+FFN | 158 | incremental native chunks |

Builder default is the combined aggressive preset; it does NOT automatically run
all six presets or launch a comparison campaign. Isolated presets allow diagnosis
if the combined case loses identity/geometry/lip-sync/stability.

158-frame plan: `[0,158),[136,294),[272,430),[408,566),[544,685)`.
All targets remain on17n+5 and >=124frames; no minimum-duration guard is removed.
Five windows use40 forwards total versus24 for the control. Lower per-forward
sequence length may help attention cost, but extra guide/reference/forward overhead
can offset it. This is a measurement, not a promised speedup.

## FP8 implementation and dependency

`acceleration_experiment.py` selects backbone attention/FFN **base** `nn.Linear`
modules, including PEFT `.base_layer`, but excludes LoRA A/B, refiner, modulation,
norms, input/output projections and VAE. No fusion or weight export.

- Dynamic FP8 activation + FP8 weight via TorchAO config version2, default per-tensor.
- BF16 named adapter tensors must be present; hashes before/after quantization must match.
- All selected weight shapes must be divisible by16, and all must actually convert
  to Float8Tensor. TorchAO's possible silent skip is treated as failure.
- A small selected-layer probe must execute `aten::_scaled_mm` and produce finite
  output. This confirms that probe, not a proof of every full-model CUDA kernel.
- Fast accumulation is explicit/configurable and recorded. No automatic torch.compile
  or calibration/model substitution. GPU performance and numerical quality need testing.

Pinned runtime: Torch2.8.0+cu128 + TorchAO0.13.0, exact2H200/SM90.
Wheel `torchao-0.13.0-1-cp39-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`:
SHA256 `718568b158af481fdd6220165d3e336b7789ad8900ac40a07bf2212505e84491`.
The7MB PyPI wheel was downloaded and SHA/namespaces inspected on CPU; not installed
on the Mac. `prepare_h3_fp8_runtime.py` installs this exact wheel **only** in the
experimental Job's restored venv, using no-deps/no-index/force-reinstall. Torch,
Diffusers, Transformers and PEFT versions must remain unchanged. No base dependency
upgrade or mutation of the control Job's environment. Install/import checks happen
before expensive pipeline preparation.

## Incremental native VAE decode

`incremental_decode.NativeDecodeStream` follows the exact pinned Diffusers native
chunk/blend protocol, not independently decoded RGB clips. Source SHA:
`e59dc54caebff9ecb6d88c957b1ff189bf6a88419ed3551f84d3f68c20ce3192`.
Apache-2.0 source attribution is retained in the implementation.

- Five latent-frame stride, seven-frame decode window, two-frame latent lookahead.
- Preserve the native overlap and cross-fade. Emit17-frame body chunks and final5-frame tail.
- Final5n+2 latent grid required; no speculative padding or final-frame rewriting.
- Bounded persistent latent/tail state, explicit finalize/concurrent-push/stale reuse guards.
- Renderer denormalizes and transfers5 latent frames at a time, encodes RGB incrementally
  with the existing PyAV/libx264/CRF18 settings, verifies685 decoded/683 exported frames,
  then muxes the full original audio ONCE losslessly using the existing PCM checks.

The dedicated CPU test extracts the actual pinned upstream `_decode` and `_blend`
functions by AST. A deterministic synthetic non-causal `_decode_clip` is substituted;
streamed concatenation is bit-identical over several final lengths and input chunk
sizes. **This is protocol parity, not testing the real VAE weights or CUDA pixels.**

This renderer still starts AFTER the whole denoise phase and writes a final MP4.
It does not overlap DiT and VAE, consume latent windows live, guarantee client
playback/physical flush, or provide future-audio causality. Full-audio conditioning
and accumulated denoise latents remain offline. First-decoded-RGB and first encoded
packet are internal timing metrics, not end-to-end time-to-first-displayed-frame.

## Separate Job building / execution

```bash
PYTHONPATH=. python3 -m research.build_h3_acceleration_experiment \
  --preset fp8-short-incremental --total-seconds 3600
```

Without `--output`, this is plan-only: no network, models, installation, reservation
or Job launch. To build a real spec, provide a reviewed native baseline spec, NEW
runtime mount/repo/revision, actual fresh ledger bound, and an exclusive output path.
Total allocation is explicit2400..4800seconds (default3600), including bootstrap;
use the actual terminal/reconciled ledger when selecting it. The existing baseline
runtime cannot be reused: launcher requires the new experimental
module closure. Use normal `launch_hf_identity_pipeline.stage/submit` with a new private
immutable deployment and fresh idle/privacy/pin/time/rate checks.

Price proof has separate scope `h3-fp8-streaming-experiment-v1`, explicitly bound to
both native and experimental contracts; baseline price proof cannot silently enable
quantization. Keep finite provider timeout and cumulative ceiling, account every
allocation, and apply the user's standing corrective-resubmission/post-report rule.
Overlapping allocation is allowed ONLY for the explicit USER-approved two-Job pair;
all other allocations retain idle/default guards. Reconcile terminal usage without re-adding archived
historical estimates. No background watcher was created.

Upload stays generated-only: review.mp4 plus continuation/decode profiles, in a new
`acceleration-experiment` run prefix. No raw reference/audio/conditioning/latent upload.
Inspect identity, facial proportions, mouth detail, temporal stability and every
boundary before calling any preset accepted.

## Sources

- https://huggingface.co/docs/diffusers/en/quantization/torchao
- https://github.com/pytorch/ao/blob/v0.13.0/torchao/quantization/quant_api.py
- https://pypi.org/pypi/torchao/0.13.0/json
- https://github.com/pytorch/ao/issues/2919
- Existing exact Diffusers source/runtime pins in `h3-native-continuation.md`.
