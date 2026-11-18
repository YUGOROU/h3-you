# Production checkpoint escrow and provider handoff

## Current implementation

`research/h3_personal/identity_handoff.py` and opt-in fields in `research/hf_identity_train.py` implement **canonical subject weights + deterministic sampling cursor** handoff. Production deployment/Modal execution are still pending. No actual remote checkpoint publication or GPU run has occurred under this redesign.

USER prefers efficient quality improvement over pursuing a715-example pass at any cost. Modal can stop at a useful recoverable checkpoint; HF is a possible continuation target. Full coverage remains measurable, not a mandatory completion/convergence claim. No automatic provider fallback or second paid run is launched.

## Escrow publication

`checkpoint_escrow=true` publishes at the initial-check boundary, configured checkpoint intervals and final graceful save. Each publication is a single private Hub commit under `identity/<run_id>/escrow/checkpoint-<step>/`, containing only:

- `adapter_model.safetensors`
- `adapter_config.json`
- `identity_receipt.json`
- `EXPORTED.json`
- `HANDOFF.json`

All canonical export integrity checks remain unchanged. HANDOFF binds all artifact digests, recipe digest, source run/local update, optional cumulative sampler cursor, and explicitly says **exported-only, no fresh-base verification, no optimizer state, no exact resume, no quality verification**. Escrow is distinct from final strict-verified upload namespace. The source stage name must match the update count. Identical local checkpoint bytes already committed are not uploaded twice. Public repositories are rejected. Offline Hub flag is restored even on error; existing training supervisor deadline bounds network work. No retry loop/background watcher. Publication failures stop rather than silently continuing unrecoverable paid work.

## Target phase

`warm_start` requires exactly `checkpoint`, `metadata`, `metadata_sha256` local-on-worker paths/pin. `research/h3_personal/identity_handoff_fetch.py::fetch_handoff` provides a remote-worker fetch helper: require private model repo and exact full commit, canonical escrow prefix and pinned metadata digest; download metadata first, bind run/update/recipe, copy only four canonical artifacts, validate every digest and atomically publish a new local input directory. No overwrite/partial tree on failure. Pipeline/launcher now wire `warm_start_remote` (exactly repo_id/revision/prefix/metadata_sha256) to this helper and forward `checkpoint_escrow`; recipe/pins are supplied by the generated training request and the Hub offline flag is restored on error. Concrete Modal bootstrap/deployment remains pending; SDK transport was mocked, not exercised against real checkpoint data. Validate metadata/artifacts/source pins/recipe before allocating/loading model tensors.

The target is a new run, with new optimizer, local step/epoch/sample counters and model RNG. It copies only canonical source artifacts into its own work root so strict reload can write a verification marker without modifying pinned/read-only source inputs. Existing fresh-base strict reload verifies exact keys/dtypes/shapes/tensor equality. A lineage receipt records the parent run/update/artifacts and explicitly denies exact optimizer/RNG resume. Full saved PEFT configuration remains required; no loading via strict=False.

Recipe binds source/model pins, frozen dataset/index, LoRA/objective, optimizer settings except requested update cap, precision, sampling mode/seed and optional conditioning/TREAD overlay hashes. Provider/work paths and new update cap can differ; learning rate/objective/sampling mode changes are rejected rather than silent. Explicitly revised recipes would require a separate reviewed migration path.

## Sampler continuity, separate from optimizer resume

`HANDOFF.sampling_cursor` binds cumulative `consumed_draws`, mode,715-draw epoch budget, seed and exact indexSHA. The new phase derives `identity_sampling_start_draw`, not restored local optimizer counters. `IdentityBatchSampler` maps this to global epoch and first-epoch offset. It yields only the remaining seeded draws in that epoch, then normal full epochs. This preserves coverage-first order and replacement draw order independently of model RNG (replacement still intrinsically repeats samples).

First-epoch sampler length is715 minus offset. Boundaries including25/714/715/716/1400 were tested against uninterrupted synthetic streams. The worker checks actual sampler cursor and publishes parent cumulative cursor + local samples_seen; singleton samples_seen must equal local updates. Old handoffs without a cursor are still weight-only and explicitly make no sampler-continuity claim. This is NOT exact training resume: optimizer/RNG are reset, and routing seed currently follows local update, not parent optimizer state.

Same pinned software is needed across providers, especially Python/random and Torch multinomial behavior; current tests use local Torch2.14.1/PEFT0.21.2, not production Torch2.8. Recipe hashes are integrity bindings supplied by a trusted deployment, not authenticated proof of unchanged live environment. A new helper/Trainer+Diffusers source closure is locally staged and hash-bound, not deployed; runtime_overlay recipe binding includes both source-overlay and pristine-package inventory hashes. Conditioning binding remains pending. The old frozen39-file bundle is not rewritten. See [Modal runtime](h3-modal-runtime.md).

## Verification and remaining work

Full suites364 tests OK (default63 skipped; existing tensor environment1 skipped), including synthetic canonical tensor reload, private atomic publication transport mocks, bad recipe/hash/flags/cursor rejection, deterministic sampler continuation and old worker regression tests. No downloaded H3/adapter tensors, raw media, real Hub writes or GPU Jobs used.

Implemented: pipeline fetch/escrow forwarding, isolated source-overlay installation with separate pristine verification import, locally staged new closure, and bounded lazy Modal SDK carrier with managed-secret reference and explicit CPU/memory/disk/deadlines. Remaining: concrete Modal bootstrap/source deployment/secret provisioning, actual pinned runtime/H3 cache verification, conditioning-adapter provenance and strict scope checks, bounded progress/quality review and same-Job previews. Carrier definition is not GPU/runtime/quality acceptance. Exact optimizer/RNG resume is not implemented; do not describe a weight/sampling continuation as lossless resume.
