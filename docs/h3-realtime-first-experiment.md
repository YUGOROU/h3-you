# Realtime-first: one isolated FastH3 experiment

## Latest authority and status

USER prioritizes realtime sequential generation over quality, permits broken
experimental output, and asks for one combined experiment. Original64, Baseline,
Turbo8 and715cache remain unchanged. This supersedes quality-first **only for this
isolated branch**, not the original Ref2VA/control contract.

Corrected Job **`YUGOROU/6ad1d351c66288145225989a` COMPLETED**. Short-run
**server throughput gate passed**, not full client/causal/personalized AV completion.

### Verified results

- Four consecutive segments,493unique frames /20.541667s,256x448/24fps.
- Warm publication interval:15.375s new video /10.042796s wall =**1.530948x**.
- Independently bounded warm GPU generation throughput:**1.523449x**; consumer3.490648x.
- Availability lateness0 for every measured segment; no growing backlog in this short run.
- Cold first publication215.274s (runner/model-load scope, excludes bootstrap); not low cold-start latency.
- Actual per-segment packet PTS, frame counts, digests, full decode independently
  verified locally. Offline lossless-remux `verified-results/review.mp4` also verified
 493contiguous PTS/frames. This assembled review is NOT live Mac playback evidence.
- Output commit `bf32e08b67bfbcff88aaa795b937a9606c5635e5`;
  review SHA256 `131e53145bcc45a455f6557cc720a5bd5faf7bb3448ebf67b5a5614d89d2e4da`.
- No external audio, client playback, long-duration stability or quality acceptance proof.
- Running524.179s ->**1.46USD** rounded-up time-rate bound (not invoice).
Private deployment/output dataset `YUGOROU/h3-realtime-first-20261016-v2`, deployment
commit `77c22656e345f166376a54a8c533cafecb1442ae`.
Local private root `$H3_PRIVATE_ROOT/gpu-hf-realtime-first-v2/`.

Initial Job `6ad1cd4e5ccb47ba0f7af8b3` ERROR:541.309s/1.51USD runtime bound.
Pinned image layer `sha256:bf314d95525961f07181e0a081a185ff97d0b24e3edfc2ba864656efa036b804`
contains dpkg status for **FFmpeg7:4.4.2-0ubuntu0.22.04.1**. Production erroneously
used `-fps_mode`, absent in FFmpeg4.4 and introduced5.1 (verified tagged upstream
`fftools/ffmpeg_opt.c`). Original stderr was not published and cannot be recovered
from terminal logs; this is an independently confirmed incompatible option.
Fix: `-vsync 0`, explicit x2642/filter1 threads, bounded stderr tail in errors,
shared production encoder exercised on first/later fMP4 segments with packet PTS
and full decode **before model asset downloads**. The new Job runs this same probe
inside its actual container. No package/model/adapter changes or unchecked retries.
GPU generation itself succeeded: first segment21.63s, second4.05s for123new frames.
One warm sample is not sustained encode/publication/client realtime proof.
**Never replay the submitted launch command.** `submission_guard.json` and shared
ledger reservation retain unknown outcomes; no background monitor/retry campaign.

## Personalized adapter: implementation started

USER reviewed the completed visual PoC: coarse resolution but coherent, without
visible breakdown, and approves moving to本人LoRA. This is visual-only acceptance,
not long-run/client/AV verification. Those checks need not block CPU implementation.

Original64 was fetched privately from its immutable escrow and SHA256 verified
(`43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151`).
Bounded public FastH3 header read plus private original header/config audit found:

- Identity416tensors/208LoRA pairs, rank16/alpha16, no rank/alpha patterns.
- All208module names and input/output dimensions match FastH3 adapter pairs;
  each overlapping FastH3 pair has rank64. No missing or mismatched pair.
- FastH3 has362pairs plus132dense tensors:82additive `.diff`/`.diff_b` and50gate
  `.set_weight` replacements, all preserved. Its pinned linear
  loader defaults absent alpha to rank. This is structural adapter agreement,
  **not** verification against every loaded T2VA base weight or identity quality.
- Runtime `set_lora_adapter` deliberately rejects changes after constructor dense
  payloads: simply adding the original adapter to the running generator is invalid.

`research/h3_personal/personal_fast_adapter.py` implements a separate **derived
constructor adapter**: preserve dense FastH3 tensors, concatenate overlapping
A rank axes and B rank axes (64+16=80). With both source scales1, the effective
update is `B_fast*A_fast + B_personal*A_personal`; no normalization/alpha guessing.
Unknown tensor kinds/scales, missing targets or dtype/dimension mismatches fail.
Original adapter/checkpoint files are not rewritten or fused into original models.

Explicit CPU tests7/7 verify summed update, dense/gate preservation, unchanged
inputs, serialization and strict coverage/rejection. Full stdlib suite569OK/84skipped. Evidence
remains private in `realtime-personal-adapter-audit-v1/` under the existing root.
### Personal GPU experiment submitted after USER `go`

Job **`YUGOROU/6ad1dc05c66288145225a2a8` COMPLETED**, private dataset
`YUGOROU/h3-realtime-personal-20261016-v1` @`6bf1ee6d3fbebe301028d995214f7f2dfc612ccf`.
Local root `gpu-hf-realtime-personal-v1/`. Never replay the submitted launcher.

- Private original83.5MB adapter/config fetched from pinned escrow; full source SHA
  checked before/after composition. Public FastH3 full5.34GB SHA checked too.
- Full builder preserves82additive+50replacement tensors; all208overlaps rank80,
  remaining154pairs rank64. Both inputs are BF16. No silent dtype/scale conversion.
- Derived artifact saved and every tensor strictly reloaded/equal; finite checks.
  Derived tensors remain private on Job filesystem, **never uploaded**.
- Explicit source overlay on pinned `lora_pipeline.py` only: original SHA
  `9db2b62897bc5f14e8cf117c5467875c41498856a6d1893c6e15a41ecab0652c`.
  Fail if any normalized key is unconsumed, adapted count is not362, or applied
  rank80 pairs not208. Both rank0/1 write required local coverage receipts before
  any generation. Profile includes receipts; missing target is not a warning-only success.
- Four256x448/24fps segments, same FastH3/VSA90%/SP2/parallelVAE/encode overlap.
  Prompt changed from toy to a white-shirted person speaking to camera. This is a
  personalized T2VA experiment, **not** isolated LoRA-only quality A/B or Ref2VA audio.
- Baseline and successful toy output untouched. No user photo/WAV input, no raw upload.
- Finite **1800s/5.00001USD** Job under latest USER scope
  `h3-realtime-personal-fastvideo-v1`, independently bound to the `go` request.
  Prior164.34incl1margin +5.00001 =**169.34001USD** ceiling bound, not invoice.
  Live idle/Modal tasks0/PRO/private permissions/price/pins/remote hashes checked.
- Full suite569OK/84skipped; explicit CPU personal tests7/7 including full-count
  serialization/dense-gate preservation and coverage rejection. GPU efficacy,
  identity/quality, client playback and causal audio remain unverified. Results below.

### Personal results verified

- Output @`13e3b4d5cc1e9280fc0c0567648af5117f90dd24`; bothrank0/1 receipts
  verify208personal/rank80 targets,362total adapter targets, zero unmatched.
- Derived SHA `507ed3067232f51682dc3b1631de6452aadcfeff2f86fe4b2c3bca8388ad172a`,
  strict CPU serialization reload passed. Original sources unchanged.
- Foursegments493unique frames,20.541667s. Remote hashes, all actual packet PTS,
  full FFmpeg decode and offline lossless-remux review independently verified.
- Warm15.375s /9.996106s publication wall =**1.538099x**; GPU lane**1.525763x**.
- **Strict throughput gate FAIL**, not end-to-end realtime completion: availability
  lateness `[0,0.512045,0,0]`. Segment1 encode0.327s/publication3.040s versus
  publication1.119..1.275s for later segments. Publication intervals5.679,1.359,2.958s.
  The one missed deadline then recovered; `growing_backlog=true` means *any increase*
  under this conservative gate, not evidence of indefinitely increasing backlog.
  Do not relax/relabel the recorded gate because the average exceeds1.
- Cold first publication215.758s in runner scope (excludes bootstrap).
- Local `gpu-hf-realtime-personal-v1/verified-results/review.mp4` SHA
  `a1c3b84a08170e16d310f5152cbec3fd73032cc71b7a416033dd34b192567bb6`.
  USER review: **identity FAIL**, unrelated person in supplied screenshot. Applied
  tensors do not establish effective identity transfer. Ref2VA→T2VA transfer and
  missing reference conditioning remain candidate explanations, not proven causes.
  USER-requested native Ref2VA implementation: `h3-fastvideo-ref2va.md` (CPU-tested,
  not yet GPU-validated; does not reuse the T2VA four-step adapter).
  Offline joined review is not live playback.
- Running524.158s ->**1.46USD** rounded-up time-rate bound. Replaces5.00001USD
  reservation; cumulative**165.80USD** incl1USD margin,4.20USD unreserved. Not invoices.
- No automatic new Job: next diagnose publication jitter/client transport and
  appropriate bounded initial buffering, without claiming long-run stability from4segments.

Code: `prepare_h3_realtime_personal.py`, `build_h3_personal_fast_adapter.py`, plus
opt-in wiring in existing realtime bootstrap/runner/launcher. This transfers
Ref2VA-trained identity updates onto T2VA; do not claim semantic equivalence.

## Combined configuration

- FastH3 four-forward distilled adapter + VSA90%, tile64.
- Two H200s as **one SP2 group**, TP1; not two independent streams and not SP2×TP2.
- 256×448 /24fps,124frames per segment, four consecutive segments; no discarded warmup.
- Last generated RGB frame conditions the next segment. Trim exactly one reused
  frame after the first:124+123+123+123=493unique frames /20.541667s.
- Parallel VAE decode (`gather`). DiT stays GPU-resident; auxiliary encoders/VAE
  offload follows upstream recipe. Compile disabled to bound cold-start expense.
- One bounded CPU encode/publication consumer overlaps next GPU generation;
  max one waiting segment, no replay/drop, owned CPU frame copies and failure propagation.
- Silent generated fMP4 segments published to a **private** Hub dataset as they
  complete; no private source/reference/conditioning/latent uploads.

This deliberately skips broad candidate benchmarks and a separate paid smoke job.
Runtime preflight, model load, all four segments and teardown share one finite Job.
It does **not** blindly combine incompatible alternatives:

- Generic FP8 plus FastH3 hybrid/full-rank delta fusion is not established; BF16
  is explicit, not a silent FP8 fallback. No claim FP8 is applied in this Job.
- RAVEN/TaoMate KV, another decoder and distributed TP are not drop-in additions
  to this runtime's model/conditioning contracts; no pretending their gains multiply.
- Native Ref2VA本人LoRA, external audio/Irodori and original recorded WAV are
  **not inputs to this visual-only dummy-prompt PoC**. Generated model audio is
  discarded. Original audio remains untouched. Not personalized AV streaming.
- Decode returns a whole124-frame segment before the CPU encoder receives it.
  This is segment-incremental delivery, **not** chunk-major causal diffusion or
  a250ms neural generator. Mac live playback/OBS/Discord are not yet connected.

## Exact pins and provenance

| Component | Pin |
|---|---|
| FastVideo code | `ba371e33ed080e51f1884ae166baf7c4447a66fe` |
| Base MiniMaxAI/MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| FastH3 adapter repo | `f509e629374cac104e7f62daecce6d1488a3041d` |
| `vsa-datafree/adapter_model.safetensors` SHA256 | `42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470` |
| Official amd64 image | `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:34ef8907bf0c4329d81be5fd4a6e3056b806e65fa0bba439733ab8a45bef31cf` |

Image is CUDA12.6.3, built from `9f595199590e7a1848a0669cbbd2b18081c321d8`.
GitHub comparison to the selected source shows **only docs/agent metadata/tests**
changed, not runtime/dependency files. Runtime source overrides its editable source
with the pinned tree. No pip installs/upgrades in the GPU Job. Early gates require
Torch2.12.0, FastVideo-kernel0.3.5, quack0.5.3, exactly2H200 and working CUDA on both.
Mismatch fails closed before expensive weights; GPU compatibility remains measured,
not inferred from image labels alone.

CPU preparation fetches only code/text/model metadata. A full repository archive
exceeded128MiB due unrelated assets, so preparation was corrected to pack the
code tree only, verifying **every selected file's Git blob SHA** against the pinned
recursive tree. Private archive is3.35MB. No full model download onto the Mac.
GPU downloads only selected T2VA components locally and verifies byte counts and
SHA256/Git blob hashes before load. Do not substitute existing Ref2VA-only assets.

Public FastH3 card/LICENSE inherits MiniMax H3 Community License. Existing
provider-bound `operator_hf_acceptance` evidence retained and hash-verified;
execution geography remains unknown, **not** verified permissible geography.
FastVideo code is Apache2.0. Source/license texts and three existing approval
JSONs are retained privately; no credentials or personal media in the deployment.

## Measurement: no prerecorded or prebuffer loophole

- Count *new* video frames only, excluding reused heads.
- Record generation, encoding and publication completion on one monotonic clock.
- Warm publication denominator spans first→last published segment, including waits.
- Additionally require warm **GPU generation throughput** and consumer service
  throughput≥1. A delayed first upload must not create a false pass by prebuffering.
- Compute availability lateness/backlog against24fps consumption. No growing backlog.
- Cold first publication is separate and includes model loading in runner scope;
  full allocation additionally includes bootstrap/download, not subtracted from cost.
- FFmpeg is ultrafast/zerolatency, no B-frames,250ms fragments. Separate encodes
  initially reset decode time: parsed `moof/traf/tfdt` fields are offset on the
  explicit12288Hz track clock. Packet-level PTS/count validation verifies the
  actual files, not just intended metadata. No timestamp-based speedup of content.
- Server throughput gate alone **never** sets `realtime_complete=True`.
  Client reception/display, continuous playback and external-audio causality need
  separate evidence. Quality gate intentionally waived for this branch.

## Budget and tests

One new narrowly named single-Job scope `h3-realtime-first-fastvideo-v1`; no reuse
or relabeling of old pair scopes. Existing cumulative ceiling remains170USD.
Live account-wide HF idleness, all Modal environments/tasks0, PRO identity,
private write permission, exact hardware price and uploaded bytes/hashes checked.
Under shared fcntl ledger lock reserve **before** the sole API submission call.

- Prior reconciled bound162.88USD (161.37+failed1.51), including1USD ancillary reserve.
- `h200x2` rate10.00002USD/hour, **2400seconds /6.66668USD maximum** for corrected Job.
- Submission cumulative reservation was169.54668USD; completed1.46USD replaces6.66668USD
  reservation. Reconciled bound**164.34USD**, including1USD margin;5.66USD unreserved. Not invoices. Queue is not GPU runtime.
- GPU-time bound includes initialization/compile/load/decode/teardown; no idle rental.
- Local full suite:**562tests OK /80skipped**; realtime-specific29/29 including
  actual CPU fMP4 packet PTS/full decode and false realtime-prebuffer regression.
- GPU execution and short server throughput verified above; Mac playback, sustained
  long-run stability, personalized audio and quality remain unverified.

Code: `prepare_h3_realtime_first.py`, `launch_h3_realtime_first.py`,
`bootstrap_h3_realtime_first.py`, `run_h3_realtime_first.py`, and
`h3_personal/realtime_first.py`, `realtime_segments.py`.
Original candidate source review: research notes.
