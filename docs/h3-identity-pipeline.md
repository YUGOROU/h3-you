# Single-H200 identity pipeline

## Operator-managed inference stages

Explicit `operator_managed_stages: true` is allowed ONLY with `preview_only: true`.
It removes internal runtime/pipeline, prepare/handoff/verification/upload and preview
subprocess watchdogs/reserves; processes wait without a phase deadline. SIGTERM,
cancellation, nonzero exit handling and all hash/base/territory/privacy gates remain.
Train workers reject this mode; it is not an unbounded-training permission.
`MODEL_VERIFY_BEGIN/DONE` identifies progress through the pinned58-file working set.

This does NOT remove HF's provider timeout or project150USD cumulative reservation
gate. Protocol seconds fields remain finite but do not kill individual stages in
this mode. The provider timer is the final execution limit; never claim a completely
unlimited Job. Operator controls manual stop, no automatic watcher/retry.

Live retry `6acee4095ccb47ba0f78fa93` submitted with provider3500s, same original64,
secondary reference/new head-turn-smile prompt/seed314159 and private assets mount.
Runtime revision4c7390b51d9901e990e33214255e826168ecc04d. Job now COMPLETED;
private preview revisionb8990a420cd5a6f4a5debbd00bd031794ffc037e, denoise785.622s.
User subjectively rates face~85%/scene consistency~95%; not objective metrics.
Latest local source adds private prompt conditioning cache and separate generation/
decode profiles; it is not part of that immutable completed runtime. See
`personalized_h3_identity_acceleration_streaming_decisions_2026-10-14.md` for pins,
cache integrity rules, implementation status and remaining GPU equivalence gates.

## Bounded inference-only regeneration

`preview_only: true` requires immutable `warm_start_remote` escrow, validated `cache_reuse` and explicit `inference` cases. Preparation preserves source/model/cache/territory checks; `handoff` fetches the pinned adapter without invoking an optimizer or `_train`. Fresh verification/private upload precedes preview. Inference-only reserves600s for verification instead of the training path's1200s; preview remains separately bounded. Existing training defaults are unchanged.

`inference.seed` selects a uint32-range seed (default42); `inference.prompt` optionally replaces cached text with a new nonempty prompt of at most2000 characters. The Modal bootstrap fetches the selected sample's small pinned reference image remotely. Prompt encoding uses the pristine H3 Qwen Ref2VA labelled image presentation, then unloads Qwen before constructing the transformer; no cache index/tensor is rewritten. The selected sample provides the existing attested reference/audio latents and target geometry. Audio remains recorded/frozen, not generated. Generation metadata records prompt hash/reference image hash/seed/final export marker hash; it does not certify quality or held-out evaluation.

For new conditions, change sample/reference association, prompt and seed explicitly. A paired comparison across checkpoints on the SAME new conditions remains necessary to attribute quality changes to training; changing conditions alone tests flexibility, not causal improvement. Preserve earlier previews/checkpoints and use a new run namespace.

The Modal carrier only forwards allowlisted phase and numeric denoise progress; arbitrary stdout/stderr, secrets, prompts and source paths are not exposed. The HF launcher explicitly allows `preview_only`; its bootstrap resolves and pins the selected reference image from the existing read-only input dataset mount. Use provider-specific approval evidence: an earlier HF operator acceptance of unknown execution country is neither a country observation nor a rights-holder license waiver. Allocate cold preparation time explicitly; fetching the entire model/cache on a new worker still occurs unless separately validated persistent storage is implemented. Private approvals/budget/one-shot markers are still mandatory. No fallback/retry or second concurrent GPU.

`research/hf_identity_pipeline.py` is the remote controller, **not a Jobs submitter**.
Default invocation checks the plan offline; it never downloads models or starts a
job. Original production Job `6acd5f9fdd8177968b435500` completed715-sample preprocessing and3 verified initial updates, then was canceled at USER request to allocate a full fresh training budget. Training-only + same-Job preview Job `6acd9c98b0ad3e9da5ea2056` is submitted; its results remain unverified.

## Completed-cache training and same-Job previews

Optional `cache_reuse` contains `repo` (same private cache repo), immutable40-hex `revision`, `prefix` (`components/<planSHA>`), and independent `index_sha256`/`report_sha256`. Preparation downloads only that frozen remote cache, requires a complete merge/actual-loader receipt, validates715 ordered frozen IDs/pools and actual component hashes/tensors, and compares encoder/model/runtime identities. The patched training loader still checks every715 sample. No media/reference/text/merge encoders execute on this route; original source media need not be copied or reprocessed. Trainer/model pins are unchanged.

Optional `inference` requires cache reuse, `sample_ids` (one or two unique frozen IDs), `schedule_points:50`, and `seconds` (600–3600). This allowance is reserved **before training** in addition to1200seconds for final fresh canonical reload/upload. After adapter verification/upload, `hf_identity_preview.py` strictly loads the final canonical adapter into a fresh transformer, reuses cached conditional text/image/clean audio, generates124frames (~5.17seconds), then decodes in a separate sequential process. Transformer and VAE stages use the SAME Job's already downloaded model directory; no second Job, fresh model download, new TTS or concurrent second GPU model is created.

The MP4 muxes the existing frozen sample's recorded audio, not reconstructed/generated audio. Samples are training-set diagnostics, **not held-out/generalization/likeness acceptance**. Output is private MP4 + generation receipt only; no latent tensors/raw media/local paths/credentials are uploaded. Inference failure does not delete the separately saved verified adapter; the whole driver still reports failure rather than pretending all requested stages succeeded. The current configured preview is one sample with2700seconds reserved.

Training emits safe `TRAIN_UPDATE` step/cap/timing/numeric loss logs; previews emit stage and denoise counters. Tests cover control flow/conditioning contracts, not actual H3 generation speed/quality. Original3-update checkpoint remains available; training-only route starts a fresh run, without optimizer/counter resume.

## Operator contract

Use an externally provisioned, immutable Linux environment and the audited trainer
source checkout (`7a1563a267a17da898df3396acbd6a9d24fcb142`). The driver does not
install packages or clone arbitrary trainer versions. Supply an independently
reviewed runtime lock, byte-pinned in the request, with expected source versions;
do not generate a purported trusted lock from arbitrary installed packages.
The encoder verifies loaded module bytes, package versions, trainer file pins and
FFmpeg executable/PATH. Checkpoint installer verifies exact source contexts and
composes guidance, v2 cache, sampling and checkpoint hooks.

Request JSON (`h3-identity-pipeline-v1`) fields:

| Field | Required value |
| --- | --- |
| `output` | Absolute new remote directory, exclusive to this run |
| `flavor` | `h200` (exactly one visible H200) |
| `overall_seconds` | Normally `21600`; at most `23400` (6.5h) |
| `stage_seconds` | Positive integer, at most `10500`, e.g. `9000` |
| `rate_usd_hour` | Operator-verified current rate, positive and <=10 |
| `spent_and_reserved_usd` | All prior spent/reserved compute; total reservation plus this run <=150 |
| `max_updates` | Explicit positive integer; no unbounded training |
| `checkpoint_interval` | Optional positive integer; default25 |
| `source_root` | Remote audited original trainer checkout |
| `source_revision` | `7a1563a267a17da898df3396acbd6a9d24fcb142` |
| `model_repo` | `MiniMaxAI/MiniMax-H3` |
| `model_lock` | `{ "path": "/remote/model-lock.json", "sha256": "<independent SHA256>" }` |
| `runtime_lock` | Same path/SHA256 structure |
| `runtime_expected_versions` | Independently reviewed exact package-version map, equal to runtime lock `versions` |
| `runtime_provenance` | Nonempty operator provenance identifying reviewed source/environment, not installed-state inference |
| `runtime_source_revisions` | Exact `identity_checkpoint.SOURCE_PINS` map (`trainer`, `h3`, `diffusers` commits), independently reviewed |
| `approval` | Path/SHA256 of approved compute evidence document |
| `territory_approval` | Trainer-compatible approval object described below |
| `transformer_files` | Explicit complete list of `transformer_ref/` configs, index and all safetensor shards |
| `output_repo` | Already-existing private HF model repo for strict-verified adapters; write-capable secret token on remote only |
| `cache_repo` | Explicit already-existing private HF **dataset** repo for component snapshots, different from `output_repo`; content-write access required |
| `input_documents` | Pinned document map described below |
| `production_inputs` | Remaining arguments for `production_inputs.stage`, described below |

Model lock schema `h3-local-model-lock-v1`: revision must be
`42ed227ee7df40d41602854ae760620d6eb651fe`; `files` maps exact encoder inventory
(`model_index.json`, all files under `vae`, `audio_vae`, `text_encoder`, `tokenizer`,
`processor`) to independently approved SHA256s. Do not put transformer assets in
this encoder-only lock: list them in `transformer_files`. Provisioning uses
`snapshot_download(local_dir=..., revision=<commit>, allow_patterns=<exact subset>,
max_workers=1)` on remote **after** approval and private-repository access checks.
Before model metadata/download, the driver checks both repositories with explicit
repo types (`cache_repo`: dataset, `output_repo`: model), requires `private is True`,
and calls `auth_check(..., write=True)` for each. Missing, public or inaccessible
repos fail closed; no repo is created and no visibility is changed. Use an SDK
supporting that write-access check; older incompatible SDKs fail closed too.
The offline plan check does not contact the Hub or establish private/write access.
Regular files/hardlinks are accepted;
symlink roots/assets and unexpected files are rejected. Local download metadata is
removed. Every asset's size and actual LFS SHA256 or Git blob SHA1 is compared to
revision-specific API metadata. Encoder files additionally match operator hashes;
the complete actual SHA256 inventory, including transformer assets, goes to the
trainer. API metadata is not a substitute for hashing local bytes.

Runtime lock schema `h3-encode-runtime-v1`: `versions`, `modules` (module name ->
SHA256), `ffmpeg: {path, sha256}`. Required encoder modules are listed in
`hf_identity_encode.REQUIRED_MODULES`; include actual text/audio/video implementation
pins. Required version entries include torch, diffusers, transformers, safetensors,
Pillow, numpy, accelerate, peft, huggingface-hub. Audited source expectations include
Diffusers `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`; operator must supply real
module hashes and versions for that reviewed environment, not placeholders.

### Compute approval (two worker formats, same evidence)

`approval` pins a document with:

```json
{
  "schema": "h3-compute-approval-v1",
  "execution_country": "<actual non-excluded ISO country>",
  "approved": true,
  "approved_by": "<reviewer>",
  "license_allows_execution_country": true,
  "country_evidence": {"path": "/remote/country-proof", "sha256": "<hash>", "description": "<provider compute-country statement>"},
  "license_approval_evidence": {"path": "/remote/license-approval", "sha256": "<different hash>", "description": "<reviewed license authorization>"}
}
```

`territory_approval` in the request contains `compute_country` equal to that
execution country, `approved: true`, `approved_by`, `evidence_kind` equal to
`provider_compute_country_statement` or `rights_holder_execution_authorization`,
and `evidence_file`/`evidence_sha256` exactly equal to `country_evidence` path/hash.
Both proof files must exist on remote and hash correctly. These are operator
attestations, not authenticated geolocation or legal opinions. No country is
invented by the driver. Collection membership, official hosting, storage-region
settings and an official Space alone do **not** establish compute authorization.
Missing/bad approval yields plan-only/readiness false, even when execute is asked
(exit2 for execute); no model provisioning begins.

### All715 input documents

`input_documents` contains exactly the stage document arguments, each as a
path/SHA256 object:

- `selection_path`: fixed frozen selection (byte SHA256
  `17900bcdcbc5a6e2441b95b824d5d7778fdfd11a461bdbc3b4dc7f63ceacd634`).
- `manifest_path`, `summary_path`: completed pinned raw production outputs.
- `source_map_path`: original MOV SHA256 -> remote relative path map (all six).
- `reference_mapping_path`: approved `h3-production-reference-mapping-v1`,
  one `{path,sha256}` reference per frozen ID, all715.
- `caption_mapping_path`: approved `h3-production-caption-mapping-v1`, explicit
  sample/session scope covering every frozen sample/session.

`production_inputs` supplies absolute remote `source_root`, `media_root`,
`reference_root`, `code_root`, and independently pinned `expected_input_revision`
(the raw producer's input revision, not a summary-inferred revision). `code_root`
contains the three flat pinned producer files required by `production_inputs`.
Media/source/reference bytes are verified through the existing stage API. Ordered
715 IDs/ranges/pools are preserved, identity590/motion125. No photo/caption is
selected automatically. Inputs must already be mounted/provisioned externally;
only the H3 model subset is downloaded by this controller.

## CLI (no submission)

Compute the SHA256 of the final request document externally; never modify it
mid-run. With the code available in the remote working directory:

```sh
# Safe offline plan check; this is also the default on a Mac.
python -m research.hf_identity_pipeline \
  --request /remote/pipeline.json --request-sha256 "$REQUEST_SHA256"

# Execute ONLY inside the externally approved single-H200 Linux Job.
python -m research.hf_identity_pipeline \
  --request /remote/pipeline.json --request-sha256 "$REQUEST_SHA256" \
  --execute-remote
```

A separately authorized detached HF Job must have a finite scheduler timeout,
no automatic retries, one H200 and a confirmed <=$10/hour rate. Set its timeout
slightly above the controller deadline, but <=6.5h total, and include startup time
in the operator's reservation. Do not launch a separate paid CPU watcher or agent.
This controller runs entirely remote after launch, irrespective of Mac sleep or
agent disconnection. It cannot guarantee completion within the deadline.

## Execution and failure semantics

1. Hardware check; runtime/source validation, composed loader install, all715
   `production_inputs.stage`, approved pinned model provisioning and actual
   inventory; computed encoder identities and composed loader file lock.
2. Encoder subprocesses: media rank0, media rank1, references, text, merge, serial
   on the same GPU. Separate processes unload VAEs before Qwen. Each encoder CLI
   receives document SHA256 pins, local model/source/loader roots, a finite
   deadline and explicit `--snapshot-repo <cache_repo>`, including merge. Merge
   invokes the actual composed v2 Dataset loader over all715.
3. Checkpoint installer prepares a fresh runtime; existing trainer `_train` API
   runs exactly one production optimizer loop with the initial1–3 finite checks
   (configured3) inside that loop, not a separate pilot/reset. Maxupdates is fixed.
   Each training request includes an explicit safe, unique `run_id` (`h3-` plus
   UUID4 hex), independent of reusable component identities and local output paths.
4. Existing `supervise(..., 'verify-checkpoint')` performs fresh canonical PEFT
   reload in a separate process and uploads only strict-verified canonical files
   to an already-private repository. Quality/likeness/sync remain unverified.

### Persistent component snapshots

The existing encoder uploader runs inside each stage on the same Job, not a paid
watcher. Its private dataset prefix is `components/<semantic_plan_sha256>/`, where
`hf_identity_encode.semantic_plan` hashes the verified plan with relocatable source
paths removed. This is **not** the request hash, output directory, run ID or raw
input revision. Reusable component IDs remain byte/recipe identities, not sample
IDs or paths; changed Reference/text does not redefine compatible video/audio.
No runtime proof or legacy-cache certification is manufactured by the driver.

Completed objects upload incrementally in bounded commits before the stage receipt
and report; interrupted uploads without a matching receipt are not a completed
stage. Allowed private artifacts are `objects/...`, `media-0.json`, `media-1.json`,
`references.json`, `text.json`, `index.json` and **`report.json`** (the encoder report).
The uploader rechecks privacy before each commit. Model weights, source files, raw
media and optimizer state are not in this allowlist. `pipeline-report.json` is a
separate local controller artifact, not the encoder snapshot report. Persisted
snapshots do not imply automatic restore/retry: this driver still requires a fresh
exclusive output directory and does not automatically certify/reuse old caches.

One remote parent process-group watchdog limits every stage and the overall run.
Preparation/encoding/training cannot consume the final20minutes reserved for
reload/upload. Training is bounded by min(21600, remaining minus1200) seconds;
verification has at most1200seconds. The full allocated training wall budget is
passed unchanged to the child: its own300second save guard is inside that budget,
not deducted a second time by the controller. Worker requests are byte-pinned by the driver
wrapper (trainer CLI itself has no `--request-sha256` flag). This uses the actual
workers/installers, never old `process_dataset` or duplicate encoding logic.

On any failure/timeout: stop, no retry or automatic continuation, readiness false.
A killed training run may leave an exported checkpoint; it is **not** claimed
valid and is not automatically uploaded. Graceful wall-stop/final-save handling
belongs to the training worker, not this driver. The controller does not require
that a wall-stopped worker reached `max_updates`: it uses the worker's final
snapshot for fresh verification and still requires the strict-verified upload
status. No usable partial adapter is promised by controller/CPU tests.
Outputs are private directories0700/files0600. `pipeline-report.json` is one bounded
snapshot with stage/failure type, never exception messages, tokens or raw media.
Worker-local diagnostics may contain paths; retain them privately. Readiness true
means the pipeline reached strict reload/private upload, not the research C gate.

## Remaining launch blocks / validation

Real provider compute-country/license evidence, final all715 approved reference
and caption maps, external pinned runtime/environment, complete model subset and
independent lock hashes, private repository/secret access, current hardware price,
and cumulative budget reconciliation must be provided before launch. No real H3
GPU execution, throughput/memory fit, perceptual quality or unattended successful
completion has been established by these CPU tests.

```sh
python3 -m unittest discover -s tests -p test_h3_identity_pipeline.py -v
```

Tests use fake SDK provisioning, private/write-access failures, every encoder
command's dataset snapshot target, actual source/model pin checks, unique train
request IDs and serial worker commands/reserves, plus a real CPU subprocess
watchdog; no GPU models, downloads, Hub calls or paid calls. Mock documents and
mock success statuses are test fixtures only, never runtime evidence.
