# Frozen production raw-media preparation (CPU, no H3)

## Scope and status

**Production deployment:** all six originals (11,281,926,381 bytes) uploaded privately; CPU Job `6accfc3b134c60eb026097d6` COMPLETED,715/715 and final complete flag true. Output revision `714cb0a67dd343c96276ae930ccd1029f9ebddb9`; ordered frozen715 metadata/pools checked from small pinned manifest, no media download. Repo `YUGOROU/h3-production-715-raw-20261012`, input revision `851c993d1b460828160a04ddf977d6fe368cdf2b`,3h hard timeout,$0.03/hour,$0.09 reserved maximum; terminal cost estimate~$0.010584, not invoice. Reservation retained until billing reconciliation. No H3/GPU/models. The latest verified state is tracked locally; this count isn't completion.

Initial API submission returned BadRequest with no Job created. After an account-wide terminal-state check, one explicitly reviewed correction retained the digest, normalized DockerHub spelling, and supplied short name/labels; it succeeded. Precise original rejected field is unproven. No automatic retry/reupload/watch loop. `reconcile_hf_raw_submission.py` is an exclusive one-attempt repair, **not** an instruction to rerun it; the reservation is already submitted.

`research/hf_prepare_identity_media.py` and
`research/launch_hf_prepare_identity_media.py` implement a separate, bounded raw-media
step. They never load H3, weights, reference photos, latent caches, or training code,
and never launch another job/GPU. This implementation task **submitted no jobs,
accessed no original recordings, and made no cloud calls**. The synthetic FFmpeg
test does not prove production throughput, complete-clip quality, speaker identity,
or audiovisual synchronization of the originals. Real production acceptance remains
unverified until all 715 outputs pass in the CPU job.

The input is `frozen_selection.json`, schema `h3-production-selection-v1`, built by
`research/h3_personal/freeze_dataset.py`: exactly 715 ordered ranges, 590 identity +
125 motion, 3694.166667 seconds including overlap, six source SHA256s,
`selection_frozen=true`, and `source_bytes_verified=true`. Both launcher and worker
rebuild the frozen semantic digest using the pinned freeze helper. Original paths
are excluded from that semantic digest; IDs, session, source SHA, quality pool and
original start/end ranges are not. All six source byte hashes are recomputed before
work, including before resume. Source-map entries must match the selection exactly.

## Local staging (default; no upload or job)

Run from the project root, with a **new** output directory outside the recordings:

```sh
python3 research/launch_hf_prepare_identity_media.py \
  --frozen-selection /absolute/path/frozen_selection.json \
  --stage-dir /absolute/path/cpu-media-stage \
  --repo YOUR_ACCOUNT/production-raw-UNIQUE_RUN
```

Staging hashes originals but performs no encoding and no original-media copies.
Only the freeze and three small code files are copied. There are no archives,
symlinks, or fallback copies of the ~10.5 GiB originals. `local_sources.json` is a
local-only direct-upload map; it and `plan.json` are never uploaded. Staging needs
only space for these small files, not the originals or processed corpus. Existing
stage directories are reused by the CLI, never silently regenerated.

Trusted flat modules are pinned by byte SHA256 in the worker:

- `freeze_dataset.py`: `e8fbb5f9be9ccc68637e38a3ae517a5ab34a5f9ff1601bea38aa1e178e59cc02`
- `preprocess.py`: `3c51ebde6529d584a0485037ff29c17112f47677e057f6f89f6eb64bb29c6206`

Changes to these helpers require deliberate review and pin updates. The staged
worker has its own byte hash in the request; deployment mounts an immutable dataset
commit and verifies code hashes before importing the flat helpers.

## Explicit submission (operator/main review only)

Submission is disabled unless `--submit` is passed. **Do not execute this merely
to test the implementation.** Review stage files, image, ledger and account first.

```sh
python3 research/launch_hf_prepare_identity_media.py \
  --stage-dir /absolute/path/cpu-media-stage --submit \
  --budget-ledger /absolute/path/reviewed-compute-budget.json \
  --image 'YOUR_TRUSTED_PYTHON_DEBIAN_IMAGE@sha256:REVIEWED_64_HEX_DIGEST'
```

Use a trusted digest-pinned Linux CPU image providing Python 3.11+ and Debian
apt/Bash tooling. Setup installs distro FFmpeg and `huggingface_hub==1.12.0`.
No model or ML dependencies are installed. FFmpeg package version is distro-defined,
not asserted to be bit-reproducible with another host's encoder.

The reviewed ledger must already exist, outside the repository, e.g.:

```json
{"spent_usd": 29.83, "reservations": {}}
```

That number is illustrative, **not current billing proof**. `spent_usd` must include
all prior reconciled compute; outstanding reservations remain separately counted.
Use one account-wide ledger and do not remove reservations without reconciling
billing and actual terminal job state. Completed reservation costs must not be
counted twice when transferring them into `spent_usd`.

Submission gates:

- Cached `~/.cache/huggingface/token` deliberately overrides `HF_TOKEN`; never print
  either token. A personal PRO account and recognized write-token permission are
  required. Fine-grained/unknown roles are rejected rather than guessed eligible.
  Successful `list_jobs` checks account access; service submission can still reject
  billing/permissions. Such errors never trigger retry.
- Personal account namespace only; all prior Jobs must have known terminal status.
  A prior job labelled with the same frozen dataset digest (full or32-character prefix) blocks duplicates. Full SHA remains in request/selection/ledger; bounded service labels don't truncate source identity.
- Hardware is exactly `cpu-upgrade`, accelerator absent, verified positive rate
  <= $0.05/hour. Hard platform timeout **3h**. Reserve `3 × verified hourly rate`
  before any upload/submission (<= $0.15; within the $0.20 CPU cap). Include it and
  all previous reservations in the cumulative **$150** cap.
- Local ledger lock, dataset-digest reservation and exclusive stage submission guard
  persist even on ambiguous network/API failures. Do not repeat the command to
  recover a failed submit. Manually reconcile first. This is not a distributed lock
  across independently maintained ledgers/accounts.
- Create a **new private dataset**, never reuse/promote a public one. Upload only the
  six known original `.mov` source paths with `upload_file`, unchanged, then the
  freeze, worker, two helpers and request. No recursive folder upload, route
  metadata, reference photos, local maps, token files, weights or latents.
- The final upload commit pins the read-only `/inputs` mount. Its revision is passed
  via `HF_INPUT_REVISION` (avoiding a self-referential commit hash inside request).
  Outputs are committed to the same private dataset under `raw/DATASET_SHA256/`.

The launcher returns a finite post-submit report with job ID, private repo, input
revision, verified rate and reserved upper bound. It does not wait, watch, retry,
or release the reservation based on estimates. No H3 territory/license gate is
claimed satisfied by this non-H3 step; that remains a separate prerequisite for
future H3 execution.

## Worker contract and snapshots

There is **one** sequential FFmpeg encoder (below the maximum of two). It uses the
existing pinned `preprocess.command` and `preprocess.validate`, without changing
selection or silently dropping clips:

- Video: 124 frames, 24 fps, 768×1344, autorotation, aspect-preserving fit and black
  pad (no crop/stabilization), CPU libx264/yuv420p, no retained original metadata.
- Audio: shared input/seek origin, stereo PCM16 WAV at 32000 Hz, aligned audio grid
  **165600 samples** (5.175s; the helper's audio grid deliberately differs slightly
  from 124/24 video seconds). Missing audio is an error, never silence substitution
  for an absent track.
- `validate` checks frame count/geometry/clock/rotation and exact WAV grid, plus full
  video decode. It is not a perceptual/content-level sync test.
- Each encode and validation subprocess has a **180s** timeout; timeout/global
  alarm kills its entire process group, including FFmpeg children. Worker wall-time
  deadline is **10500s**; platform timeout bounds setup and job at 3h. Encoding
  throughput on real CPU-upgrade is unknown; 3h completion is not promised.
- Temporary clip outputs are cleaned on error. Exceptions stop immediately with
  nonzero exit, no skip/retry, no final success. Remote snapshots remain explicitly
  incomplete unless the final atomic upload succeeds. Error reports include only
  exception types, not possibly credential-bearing SDK messages/stderr.

`manifest.json` is an ordered `h3-production-raw-media-v1` envelope with `samples`.
Each sample preserves all frozen fields, relative video/audio paths, SHA256 of both
output byte streams, explicit source-hash/range provenance and measured preparation
seconds. `summary.json` records freeze/code pins, immutable input revision, elapsed
worker seconds through checkpoint construction, processed count and completeness.
Training readiness remains false even when raw preparation completes.

Private atomic snapshots upload only dirty clips plus manifest and summary at every
10 completed clips or 300 elapsed seconds, checked between clips; a running encode,
validation or upload may extend that interval. An initial incomplete snapshot is
written before encoding, and the final success snapshot requires all **715** to
have passed. There is no continuous uploader/watcher. A failed upload aborts; its
outcome may be ambiguous and must be reviewed, not automatically retried.

Resume is **explicit** (`hf_prepare_identity_media.py --resume` in an operator-
reviewed bounded execution of the same immutable input revision). The launcher
intentionally does not automatically submit a second job for the same dataset.
Remote resume pins the current output commit, verifies the ordered frozen prefix
before fetching media, checks input revision, rehashes every restored MP4/WAV and
runs the existing validator again. It resumes only after that prefix; missing or
corrupt snapshots fail closed. Nonempty output without `--resume` is rejected.
Resuming real paid execution requires manual duplicate/ledger reconciliation;
this task authorizes no automatic recovery submission.

## Tests

```sh
python3 -m unittest discover -s tests -p test_h3_prepare_identity_media.py -v
```

Tests use six tiny fake originals, 715 synthetic frozen rows, fake Hub APIs and
snapshot sinks. A six-second lavfi fixture exercises the actual pinned FFmpeg
command/validator, portrait fit-pad, exact video/audio grids, original-byte
preservation and corrupt-output rejection. The test skips only if FFmpeg/ffprobe
are unavailable. No production media, cloud, token cache or paid calls are used.

Actual verification in this task: **16/16 new tests passed**, including the real
synthetic FFmpeg fixture (not skipped). Full stdlib suite: **144 tests, OK, 19
optional tensor/ML tests skipped**. Job deployment and real CPU throughput remain
unverified; main/operator review and explicit submission are still required.
