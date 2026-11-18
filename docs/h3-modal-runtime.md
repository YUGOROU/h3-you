# Bounded Modal carrier and isolated H3 runtime

## CLI read-only progress stream

Discarded child stdout/stderr cannot be recovered. `research/stream_modal_identity_progress.py` streams **state snapshots**, not original logs or per-update loss, from an existing container without stopping/restarting the Job:

```bash
uv run --with tqdm python research/stream_modal_identity_progress.py ta-CONTAINER_ID \
  --modal-cli "$(command -v modal)" --interval 15 --seconds 600
```

Use `modal container list --json` to identify the current container. Defaults are15s interval/10min total; maximum duration1h, minimum interval5s. Default display uses local-only tqdm for cache/model inventory bytes and saved checkpoint/escrow steps; unknown totals have no percentage/ETA. Saved-step total, when available, is the update ceiling, not guaranteed completion. Inventory may decrease on cleanup; displayed counts are not transfer throughput. Add `--json` for dependency-free raw allowlisted JSON: bootstrap/pipeline phase, runtime-ready presence, checkpoint/escrow step when saved, worker status, cache/model file counts/bytes/latest write times. Counts include partial download/metadata files and do not certify complete validated artifacts. No environment, token, recorded contents or arbitrary receipt paths are emitted. A checkpoint step is not a live update counter; it changes at saves. Small file inventories are bounded at20000 entries; no GPU/model library is imported.

Ctrl-C stops the observer CLI, never the App/container/trainer. If disconnected, the remote observer ends at its own finite deadline. It uses negligible CPU/file metadata I/O on the existing container, not another GPU Job. If the container exits, rerun only after explicitly identifying an existing container; there is no automatic Job creation or retry.

## Carrier definition and execution

`research/modal_identity_worker.py::build_app` lazily defines an SDK App/Function; it does not build an image, resolve a secret, deploy, submit or invoke a Job. Actual Modal SDK1.5.4 accepts the definition. Current bounded execution/observations and failure receipts are tracked in a local status note; an existing JP container is running preparation, with training not yet verified.

The carrier requires an exact file inventory/digests and byte-bound Python entrypoint, immutable original PyTorch image, explicit dynamic HF secret (preferred) or managed secret reference (`HF_TOKEN` required; never store a token in source/spec), finite startup/execution deadlines and bounded function exposure. It defines one H200, CPU request/limit4, memory request/limit256GiB, ephemeral disk512GiB, one maximum container, no warm pool, no configured retries, single-use containers, two-second scaledown. Serialized function definition is necessary for the lazy factory; deprecated `max_inputs` is not used. Source must contain the actual carrier module; `PYTHONPATH=/deployment` makes its explicitly copied import closure available without automatic source inclusion.

At base pricing, GPU+CPU+RAM is$6.774192/hr. The function exposure calculation includes startup and scaledown and must fit the explicitly approved remaining function allocation (initial49, reduced46.50 after a failed-run reserve; total phase50 including builds/prior attempts). **This is not a platform-wide billing cap:** image-build costs, provider billing semantics and any premium placement require separate review. Image preparation belongs to the protected pipeline allocation. Documented regional multipliers are enforced. Default <=$10/hr remains; pinned user JP/H20050USD approval permits only11.854836/hr for this finite phase. Container limits are not account-wide concurrency control across separately submitted Apps.

`research/modal_bootstrap_identity.py` consumes a byte-bound bootstrap request, validates provider-bound evidence before downloads, restores the pristine runtime, downloads only six JSON inputs and one/two selected short preview PCM files on the remote worker, and invokes the existing cache-reuse train/verify/preview pipeline. It does not submit Jobs. Child output is suppressed; a bounded local status snapshot yields only predefined phase/error-type diagnostics in the carrier result, never raw logs or paths. Quality verification is not claimed. Source/document inventory checks are integrity checks, not a scanner proving that arbitrary supplied source contains no secrets. Review the entrypoint and closure before submission.

## Credentials

The local harness `HF_TOKEN` has restricted permissions and must not be used for personal repositories. `research/provision_modal_identity_secret.py::cached_token` reads exactly `~/.cache/huggingface/token` (or an explicit test path), ignores `HF_TOKEN`/`HF_HOME`, and never prints credentials. Verify private repository access using `HfApi(token=token)`, then call `build_app(..., hf_token=token)`. The factory creates `modal.Secret.from_dict({'HF_TOKEN': token})` for dynamic injection. Token/secret objects are absent from the serialized worker function's closure; credentials are not written to spec, source or local files. Never pass a token through CLI arguments or logs.

Explicit cached credentials passed actual private input/cache/output permission checks. The named secret briefly created during setup was removed after switching to the requested dynamic approach. No App/image/GPU was run.

## New immutable helper/source closure

`research/stage_identity_redesign_runtime.py::stage` verifies an original runtime manifest and its complete byte inventory, copies immutable CPU evidence/native source pins, replaces the complete research helper closure with current code, rebuilds the pristine GPU-runtime manifest, and includes the independently pinned routing source overlay. Atomic fresh-directory publication produces `REDESIGN_CLOSURE.json`. It never modifies the old runtime, frozen39-file bundle, installed dependencies or production defaults.

The previously staged `production-715/gpu-redesign/runtime-mount-v2/` has126 files (excluding its closure document):

- runtime manifest SHA256: `49315fb62ee47571d57730bd03bc881467a7f01b4bb92f7974ab4b593094aab4`
- closure document SHA256: `3e58f958990c23adaa98fb685d4f2729eb6fcb45013922720381306c4af9c0af`
- pristine Diffusers package inventory SHA256: `d96eb712b23c545fd4a7782cd20e7f5669ab96fa7102b9478be0c46f204631ae`

The package inventory was derived from993 Git-blob-verified files of the existing pinned Diffusers revision (989 package files). Only source code was fetched locally; no weights or media. This v2 snapshot predates the concrete bootstrap and dynamic-token changes; regenerate a fresh closure before deployment. The original CPU image/dependency/interpreter evidence remains authoritative for pristine restoration. New helper bytes do not imply new GPU compatibility evidence.

## Isolated overlay materialization

`research/apply_identity_runtime_overlay.py::install` requires a pinned source-overlay manifest, independently pinned complete pristine Diffusers package inventory and exclusive destination. It atomically assembles:

- `runtime/src/h3_trainer/…`: reviewed composed Trainer sources;
- `runtime/diffusers-shadow/diffusers/…`: complete copied pristine package, with only the hash-guarded H3 transformer overlay substituted;
- `RUNTIME_OVERLAY.json`: exact source inventory and explicit no-site-packages-modification/no-GPU-verification flags.

Worker/pipeline opt-in `runtime_overlay` requires exactly `stage`, `stage_sha256`, `base_diffusers`, `base_inventory_sha256`. Paths must be absolute. Recipe binds both hashes, not provider-specific paths. Worker installation pins its generated receipt, checks every source byte before import, and loads the Diffusers shadow only in a fresh training process. Separate final verification imports pristine installed Diffusers and rejects an already loaded/shadowed process. Default baseline installation is unchanged.

**Overlay materialization is not enabling TREAD or adopting third-party conditioning weights.** Routing stays default-off; guidance+TREAD remains rejected. Conditioning provenance, scope/hook integration and a reviewed active recipe remain separate gates. Import-routing tests use synthetic package stubs, not actual H3 model inference.

## Checkpoint bootstrap fields

Pipeline/launcher now forward `checkpoint_escrow` and `warm_start_remote`. Remote handoff requires exactly `repo_id`, full immutable `revision`, canonical escrow `prefix`, pinned `metadata_sha256`. It delegates private metadata-first/artifact-verified atomic fetch, binds the current recipe/pins, restores Hub offline state on failures, then passes local `warm_start` to the existing strict worker. There is no fallback provider/second Job. See [checkpoint handoff](h3-checkpoint-handoff.md).

## Spec, placement and execution receipt binding

`modal_identity_spec.bind` checks supplied Modal placement/pricing evidence and matching country approvals, pins the closure/runtime manifest, and writes an independently byte-bound bootstrap request. Carrier and bootstrap must request the same cloud/region/country/evidence; the SDK Function receives the explicit cloud/region. Approved placement quotes must use the actual published multiplier (broad1.15x/narrow1.75x), not fictitious base-price regional selection. Single-country Modal region identifiers are checked. See [GPU choice and pricing correction](h3-modal-gpu-selection.md). Evidence is supplied and checked, never fabricated. Requested placement is not an independently observed compute-country claim.

`launch_modal_identity.submit_once` performs local approval checks with path-only rebasing (original claims/evidence bytes are unchanged), explicit cached-token permissions, forward/cumulative caps and a mandatory fresh cross-provider GPU-idle check. It writes an exclusive durable outside-closure marker before the sole image-build/submission attempt. Exceptions leave the marker reserved/unknown; no automatic resubmission. It returns the detached call identifier without a watcher.

After train/strict upload/preview, `modal_identity_receipt.publish` writes one bounded private HF commit under `execution/modal/<requestSHA>/receipt.json`; failures also publish the available pipeline status when that stage was reached. A120-second carrier reserve protects receipt publication. The receipt links the generated run ID and latest escrow commit/prefix when present, explicitly distinguishes exported-only escrow from final verification, and denies likeness/observed-country verification. Validation failures before the pipeline remain bounded local diagnostics, not claims of saved checkpoints. Receipt publication failure fails the bootstrap; checkpoints already atomically committed remain separate durable artifacts.

Current helper closure: `runtime-mount-v5`,131 files, closureSHA `e0cb15255dccfd5e53119e15cec378413acf519a2ed1e9bc9306cddfe5e86265`, runtimeManifestSHA `a7690f1c3b294e6ea74ac014b3c7d1d66270980e222ce7e57ac5d576b45da0bb`. Earlier snapshots are superseded. User Japan/H20050USD approval, provider published Japan-region/pricing statements and concrete bootstrap request are now bound. The first definition attempt failed before GPU spawn because this account/API rejects AWS cloud pinning; stopped App/tasks0 confirmed. Corrected-cloud call was explicitly submitted, then failed at first carrier validation (`closure inventory mismatch`) before bootstrap/training. Local reproduction shows importing bound Python sources creates unmanifested __pycache__ files. Image and child environments now set PYTHONDONTWRITEBYTECODE=1 before imports; fresh-process regression passes without weakening exact inventory/hash checks. Bytecode-fix closure is prepared only, no automatic resubmission. Prior failed App wall756s reserved conservatively2.50 (not invoice); same total50 now splits next function46.50/build1/priorGPU2.50, with shorter deadline. Latest receipts/ready closure/next steps are tracked locally. USER subsequently approved a separate HF functional smoke before Modal production: H200 Job `6acdff3dcc1b4162d9b9bbec` was submitted with3updates/7200s/$9.99996 upper bound, and last observed RUNNING. Its runtime/deployment are private immutable revisions; latest state is tracked locally. CUDA/training/checkpoint/preview success is not yet verified. This explicit smoke supersedes the prior no-standalone-pilot restriction only for this run; there is no automatic retry or Modal fallback.

## Still required before production

1. Use the implemented binder and one-shot launcher with existing valid placement/pricing/approval inputs and a fresh account-idle observation; no actual deployment/Job invocation has occurred.
2. Bind provider-appropriate existing evidence without reusing HF-specific territory acceptance as Modal authorization. Fail before model downloads on invalid evidence.
3. Confirm disk inventory, actual production Torch2.8/H3/cache compatibility and active training recipe; do not claim streaming/pruning speedups from source plumbing.
4. Preserve the positively reviewed HF3-step reference; retain canonical export, separate pristine reload, recoverable checkpoints and1–2 same-Job previews. Do not adopt unverified conditioning/scope changes solely to request a faster GPU.
5. Obtain advance approval for the concrete ETA>=30min plan. No standalone paid pilot, automatic retry, continuous watcher or second paid run.

Latest full local suites:372 tests pass (default63 skips; existing tensor environment1 skip). After the last receipt-linkage edits,12 targeted Modal tests also pass. Bounded HF smoke completed:3 finite updates, private canonical escrow/final reload marker and generated124-frame preview verified. USER reports strong frontal likeness/consistent scene; not held-out/generalization or LoRA-vs-base attribution proof. No local model/checkpoint/media payload downloaded. Modal remains not GPU-executed. Modal binding/launcher/transport still need execution verification; local tests are not end-to-end GPU acceptance.
