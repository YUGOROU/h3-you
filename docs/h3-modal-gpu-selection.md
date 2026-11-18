# Single-GPU Modal production choice

The single-GPU production sections below describe the earlier training scope,
not the current SMD Ref2VA SP2 inference configuration. See the final section
for the current read-only investigation of unpinned placement.

Keep one GPU and the verified BF16 base/objective/cache; no async/distributed merging, automatic SKU fallback, FP8 or quantization changes. Retain the positively reviewed HF3-step checkpoint as the reference and weight/data handoff source; optimizer/counters/model RNG restart, not exact resume.

## Current official pricing and compatibility

Sources: [pricing](https://modal.com/pricing), [GPU requirements](https://modal.com/docs/guide/gpu), [region selection](https://modal.com/docs/guide/region-selection), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

With4 physical CPU and256GiB RAM:

| Single GPU | Base$/hr | Japan region `jp`$/hr | Cost break-even vs H200 | Current deployment |
| --- | ---: | ---: | ---: | --- |
| H200 |6.774192|11.854836|1.00x|HF functional smoke passed; Modal not yet run|
| B200 |8.484192|14.847336|1.2524x|CUDA12.8 family compatible in principle, H3/backend not GPU-verified; code guards still H200-only|
| B300 |9.333792|16.334136|1.3779x|Requires CUDA13.1+; incompatible with current locked CUDA12.8 image|

B200 must reduce equal-work time by about20.2% just to break even on compute dollars (more if its setup/compatibility overhead increases). There is no measured H3 speedup or GPU-specific quality equivalence yet. PyTorch's general Blackwell support is not proof of efficient BF16 H3 head/sequence attention backward kernels. Never silently fall back to quadratic math attention for this52330-token workload. B200+ is also excluded: it may allocate B300, incompatible with the current image.

HF observations: first update87.499s, subsequent logged updates around72–73s; peak115.6GB VRAM; successful preview denoise797.034s. These are baseline observations, not B200 predictions or matched steady-state hardware benchmarks.

## Regional pricing correction

Every explicit Modal container-region selection has a premium: broad `us/eu/ap`1.15x, narrow (including `jp/ca/au/mx`)1.75x. AWS region strings such as `ap-southeast-1` are NOT the documented Modal region identifiers. A broad `ap` pool cannot be asserted to be a single particular allowed country. Routing region is not compute-country evidence.

The earlier carrier's explicit region + base multiplier1 assumption was invalid. `modal_gpu_plan.py`, spec binding, carrier exposure checks, bootstrap and launch preflight now use actual documented region multipliers and single-country region mappings. No approval documents or country facts are fabricated. The default <=$10/hr authority remains. USER explicitly approved Japan H200 up to$50; a pinned `h3-modal-rate-approval-v1` exception permits only that GPU/region/rate/cap through pipeline and trainer validation. Other higher-rate requests still fail closed.

## Recommended next production phase

Prefer the existing H200 runtime first: it is the only GPU with actual H3 functional acceptance. USER approved Japan-constrained H200 up to$50 total. Bound function14200s/startup600s/scaledown2s has$48.743134 exposure, function reservation49 plus image-build reserve1. Pipeline has200-update ceiling but stops for wall/final/preview reserves; no expected200 completion, forced715 full pass or quality guarantee. HF3-step weights/data cursor are retained; new optimizer/counters/model RNG.

B200 is a later option only after explicit guard/probe support, SM100 binary and representative BF16 attention forward/backward checks, pristine reload/preview checks, and measured speedup sufficient for the cost premium. Such checks belong to the beginning of the same approved production Job, not an automatic additional paid pilot.

First definition attempt was rejected before FunctionCreate/spawn: `Pinning cloud aws not supported.` App observed stopped/tasks0, no GPU training or FunctionCall. Failed exclusive marker and50 reservation remain; no automatic retry. Corrected closure uses `cloud='provider_selected'` internally and omits SDK cloud kwarg, while still specifying exact `region='jp'`; never claim an observed AWS cloud. This preserves user-approved Japan/H200, not a GPU/region fallback. A continuation requires a new exclusive attempt marker tied to the original50 reservation, not another50 reserve. Actual build charges remain unknown.

Bootstrap restores the pinned runtime in stock PyTorch image, then re-execs its verified interpreter before Hub imports or metadata/media retrieval. The stock image does not supply the Hub package. Current full tests377 pass (default63skipped/tensor1skipped), then16 targeted Modal tests after cloud-pin correction pass.

## Ref2VA: unpinned placement/country investigation (2026-10-17)

Initial investigation was read-only. Subsequently USER approved the bounded,
model-free probe below. No H3 inference or higher-hourly-rate exception was approved. Official runtime documentation:
https://modal.com/docs/guide/environment_variables (underscore, not hyphen).

`MODAL_REGION` is explicitly **a Modal container region name**, the narrow option
when one exists (examples `us-east`, `uk`), otherwise broad (`us`, `eu`). It is
**not a guaranteed AWS/GCP native region or ISO country code**. Claims such as
`MODAL_REGION=us-east-1` in third-party skill material are not the provider's
contract. `MODAL_CLOUD_PROVIDER` adds AWS/GCP/OCI identity, not location precision.
Both environment keys are reserved by the provider. `MODAL_TASK_ID` identifies the
container, useful for detecting repeated calls to the same allocation.

This supports only a partial fail-closed classifier: if returned metadata exactly
matches a documented single-country option `jp/au/ca/mx`, infer JP/AU/CA/MX from
that provider-defined name; other nonexcluded countries are not ruled out, but
must not be guessed. Explicit US/EU/UK names reject. `ap`/`ap-northeast` or other
non-country-specific names are **unknown**, not approval; do not load/download H3
or personal assets. The docs do not guarantee that unpinned Japan, for example,
will report `jp` rather than a larger group. Thus metadata-only screening is
possible for some reported values, not yet a universally reliable country test.

Installed SDK1.5.4 read-only protobuf inspection: `TaskInfo` has task timestamps,
GPU type/config and sandbox ID, no country/AZ/native-region field. Container args
also expose no such field. Scheduler placement has `regions` and private `_zone`,
not a documented country-exclusion selector. This is not an exhaustive claim
about private server APIs. Do not rely on `_zone` as an undocumented pricing
bypass. Do not probe provider-managed host metadata/credentials or infer GPU
location from public egress IP or routing region.

Repeated calls need not yield fresh or independent placement: functions reuse
containers and placement is a scheduler decision, not a promised uniform lottery.
SDK supports `single_use_containers=True`, which avoids reusing a function
container but does not guarantee different geography. Successful country-only
probe does not reserve that placement for a separately launched inference job;
check metadata inside the **same container that will load the model**. A CPU-only
probe's country is not evidence for later 2H200 placement. Probability of allowed
placement, cold-start/termination costs and available H200 geography are unknown.

Unpinned rates, no premium, standard Function pricing:
2H200+8physicalcores+256GiB =11.502432USD/hr (4cores=11.313792).
For metadata-only screening, 2H200+1core+1GiB nominal request=9.134352USD/hr;
60s at that usage=0.1522392USD, excluding startup/shutdown/build costs. Actual
resource use can exceed requests unless explicit supported limits are applied.
Do not claim these are hard exposure caps or proof the inference fits1GiB RAM.
Full inference still exceeds existing10USD/hr; country screening alone does not
resolve that rate constraint. No paid lottery/probe authorized by this read-only
investigation, no automatic retry loop, no unchanged region/rights acceptance.

Practical next options: ask Modal whether unpinned 2H200 can report single-country
values and whether finer supported placement metadata/exclusions exist without
region premium; alternatively a separately budgeted, model-free, finite metadata
probe. Even a short probe must include startup/teardown and reject unknown values.

### Actual model-free probe result

App `ap-HhVkjwgcYRcEALLfFGkWTo`, SDK1.5.4, requested2H200 with **no region or
routing_region**, hard1physicalcore/hard1GiB, no secrets/volumes/source mounting,
network blocked. Runtime reported **CLOUD_PROVIDER_AWS / ap-northeast**; nvidia-smi
showed two NVIDIA H200s. Country is **UNKNOWN**, not verified Japan or Korea.
Stopped after first call (max3 calls authorized); unknown metadata triggered no
blind retry. No H3 code/weights/reference/audio downloaded or loaded. App observed
stopped/tasks0 after context exit. Parent run wall7.976s, not GPU-billed duration.

Phase reserved/conservatively accounted **1USD**, not an actual1USD charge; invoice
and platform resource-billing duration unverified. Nominal request rate9.134352/hr.
New private incremental ledger `ref2va-modal-budget.json` tracks this20.44Modal
extension separately from the closed HF-stage bound; remaining bound19.44. It
must not be transferred into the HF allowance. Private receipts and exact source
snapshot: `modal-location-probe-20261017-v1/`. No submitted probe replay.

Probe implementation `research/probe_modal_ref2va_location.py` defaults to dry run;
three CPU tests passed. More paid draws are not currently useful evidence of country
unless a supported finer metadata path or a single-country returned token appears.
Metadata from this ended container cannot authorize a different future allocation.
The FA4 inference experiment itself has **not** run.

