# H3 model storage

## Current rollout

User approved one HF inference retry with a58-file volume, and a CPU/private-repo
fallback. Corrected HFv2 stopped before preview: fresh reload passed, but upload
rejected the fetched escrow's noncanonical path. Inference-only handoff now copies
unchanged artifacts to `training/checkpoint-0000064/canonical` before fresh verification.
No optimizer or checkpoint/base change. Regression exercises the failed uploader.

CPU Job `6acedb745ccb47ba0f78f753` COMPLETED (cpu-upgrade,3600s cap,<=0.03USD reservation).
Destination: private `YUGOROU/h3-ref2va-diffusers-assets-20261014-v3`.
`research/hf_curate_h3_assets.py` processes one file at a time, independently verifies
original pinned Git/LFS bytes and receipt hashes, uploads to `assets/`, removes each
local staging file, and publishes CURATED.json only after all58files plus LICENSE.
Partial upload is NOT ready for inference. This copy completed: immutable revision
`147bf61b8128a8134ceff5b30cd7b5aa3d3f5dff`, exactly58asset files/144,051,139,883bytes,
independently checked against original source and CURATED marker.
GPU retry `6acedd6ec662881452233d4a` submitted after terminal CPU success, fresh
immutable runtime and budget/idle/private permissions/replica metadata preflight.
Job ended ERROR before inference: prepare790s watchdog interrupted SHA256 read
on the mounted model. Cache715fetch/validation completed first (~211s fetching).
Total created-to-finished968.7s; rate estimate1.345411285USD, NOT invoice; existing
GPU reservation retained. failure-observation.json preserved. Server metadata did
confirm path=assets/read_only=True/fixed147bf... Volume. Persistence succeeded;
current cold mounted-read preparation did NOT finish within its allowance.
No remote handoff-copy fix/new-condition inference verification or automatic retry.
Next attempt needs explicit authority and a measured CPU mounted-read/hash timing
gate; do not guess preparation speed or bypass byte verification.
Subsequent explicitly approved v4 `6acee4095ccb47ba0f78fa93` COMPLETED with the
same immutable58-file mount after internal stage watchdogs were removed. Private
preview revisionb8990a420cd5a6f4a5debbd00bd031794ffc037e; latest denoise785.622s.
Provider3500s limit remained. Mount loading/fresh canonical handoff/inference are
now exercised successfully; no cold/warm locality/speed comparison is established.
No auto watcher/retry. No model weights downloaded locally.

No documented Volume include-list exists; SDK exposes only a subfolder prefix.
The implemented GPU launcher mounts the curated repo's `assets` prefix read-only
at `/mounted-h3-assets`. Driver verifies replica metadata against the ORIGINAL pin,
then checks the EXACT58-file local mount and every SHA before loader use. No cache
miss download fallback, symlink workaround, chmod or mount mutation. Path propagates
to encoder, train request and preview. Unit tests and completed v4 GPU inference
exercise the mounted path; this is not proof of a warm GPU-local physical cache.

HFv2 reservation reconciled to1.55USD from terminal timestamps1097.263s (NOT invoice).
CPU plus submitted4400s H200 retry cumulative conservative bound149.88082667USD.
GPU cap6.11108667USD, runtime1200/pipeline3200/verify600/preview1800seconds.
No additional paid attempt is authorized on failure.
Read-only account storage inventory:513,702,241,289private bytes; adding model plus
small bookkeeping stays below658GB and advertised PRO1TB private allowance.
No storage overage is authorized. Modal Volume remains uncreated.

## Verified assets

Hub `MiniMaxAI/MiniMax-H3` main resolved to existing production pin
`42ed227ee7df40d41602854ae760620d6eb651fe` at read-only inspection.
The linked `Ref2VA/` original-layout assets total144,051,182,613bytes.
Our pinned Diffusers/Aviad working set is58files/144,051,139,883bytes,
including `transformer_ref/` and the model-lock shared text/processor/VAE assets.
These are distinct layouts: do not replace the working set with `Ref2VA/`
or change the checkpoint base/loader without a compatibility gate.
Use a40hex revision, never runtime `main`.

## Modal

Create a dedicated private H3 assets Volume rather than repurposing/deleting
unrelated project Volumes. Read-only inventory found no H3-named Volume;
existing generic HF caches have NOT been inspected or assumed compatible.

Suggested directory structure:

```
h3-model-assets/
  minimax-h3/<revision>/<working-set-manifest-sha>/
    assets/       # ONLY the approved58files, no bookkeeping inside this tree
    inventory.json
    VERIFIED.json
```

A bounded, explicitly approved CPU-only Function in an allowed execution country
fetches assets to an exclusive staging prefix, checks every byte against pinned
Hub Git/LFS metadata and local model lock, and publishes a verified prefix only
after success. Explicit commit persists it. One writer; consumers read-only.
No secrets/source execution from unmanifested files. Model LICENSE still applies;
global Volume availability is NOT proof of storage or execution country.

GPU Functions mount the verified prefix read-only, preferably consume it directly
or benchmark copying to ephemeral SSD. No implicit fallback to Hub on a cache miss:
fail closed and request an explicit preparation attempt. Extend current bootstrap
and trainer inventory checks deliberately, never disable SHA or exact file-set checks.
A Volume is distributed storage, not guaranteed GPU-local NVMe; cold hydration,
hashing, RAM/VRAM transfer and model construction remain. Modal documentation quotes
up to2.5GB/s, not a guaranteed H3 measurement.

## HF Jobs

Mount the fixed original model repository read-only with
`Volume(type='model', source='MiniMaxAI/MiniMax-H3', revision=<pin>,
mount_path='/mounted-h3', read_only=True)`.
This is supported storage configuration. Current launcher instead integrates a
private curated model mount with `path='assets'` to retain exact working-set checks.

A whole-repo mount includes more than the approved working set and clashes with
current exact inventory checks. Implement an explicit verified selected-asset source
adapter/working view, or a private curated snapshot containing ONLY the58approved
files. Avoid duplicating original Ref2VA plus converted transformer weights.
Do not just point the existing model root at the whole repo or loosen validation.
Mounted files are fetched lazily; mounting does NOT guarantee no cross-Job network
transfer or a warm physical GPU-host cache. Measure cold/warm time and bytes.
Buckets are another option, but duplicating the model into a bucket alone is NOT
proof of better locality, speed or cost; review storage region, immutability and fees.

## Cost and rollout gates

Current cumulative compute ceiling is nearly150USD; the user-approved bounded
CPU curation/private repo and GPU retry described above have launched within
terminal-reconciled reservations; no Modal Volume has launched. Before any further
compute, wait for terminal accounting or obtain explicit revised authority. CPU time, persistent storage and
egress require separate bounds; a storage allocation continues after GPU shutdown.
Modal currently advertises1TiB/month free Volume allowance, then0.09USD/GiB-month;
144GB is~134GiB/~12.1USD/month if fully chargeable. Account-wide free allowance
availability is NOT verified. Document retention/deletion policy before creation.
Do not delete any existing Volume as speculative cleanup.

References:
- https://modal.com/docs/guide/model-weights
- https://modal.com/docs/guide/volumes
- https://modal.com/pricing
- https://huggingface.co/docs/hub/jobs-configuration#volumes
