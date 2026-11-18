# Production input staging (no jobs/downloads)

`research.h3_personal.production_inputs` is a stdlib-only bridge from the frozen
715 selection and **complete** raw-media preparation outputs to the row list
accepted by `component_encode.make_plan`. It never chooses photographs, generates
captions, changes reviewed pools, loads weights, encodes media, or starts jobs.
No suitable published reuse asset was found for this project-specific contract.

## Mounted inputs and checks

Run on the machine where originals, prepared clips/audio, approved reference
images, and producer code are **already mounted**. Do not download large media to
prepare this stage. Inputs must remain immutable throughout verification and use.

Required files:

- `selection`: `freeze_dataset.py` output (`h3-production-selection-v1`).
- `manifest`, `summary`: `hf_prepare_identity_media.py` output. Manifest schema
  must be `h3-production-raw-media-v1`; summary has no schema field in its producer.
- `source-map`: JSON object mapping the exact six frozen source SHA256 values to
  relative `.MOV` paths under `source-root`. Frozen original absolute paths remain
  provenance labels; this relocation map prevents accessing those original paths.
- Three flat producer files under `code-root`: `hf_prepare_identity_media.py`,
  `freeze_dataset.py`, `preprocess.py`. Exact supported byte pins are in
  `production_inputs.CODE_PINS`; summary code/helper pins must match them.
- Explicit reference and caption mapping files described below.

Validation rebuilds the freeze's semantic dataset SHA, ordered 715 IDs, 590
identity / 125 motion pools, geometry and durations. It checks complete=true,
processed_clips=715, summary dataset SHA, exact freeze file SHA and an independently
supplied 40-hex input revision. Every raw row must match its ordered frozen row,
including source path/SHA, session, range and pool, and carry the exact source/range
`input_provenance` proof. Actual bytes of all six mounted originals, every video
and audio file, every assigned image, and the pinned producer files are hashed.
Within a stage invocation, each resolved file is hashed at most once.

Raw paths must be exactly `clips/<ID>.mp4` and `audio/<ID>.wav`. Source/reference
mapping paths must be relative POSIX paths: no absolute paths, backslashes, empty
segments, `.` or `..`. Resolved files must stay inside their explicit roots;
symlinks that escape those roots are rejected. Missing files and hash changes fail
closed. Duplicate JSON keys and nonfinite JSON constants are rejected.

## Operator-approved mappings

Reference schema (repeat an explicit association for **every one of 715 IDs**, no
extra IDs):

```json
{
  "schema": "h3-production-reference-mapping-v1",
  "approved": true,
  "approved_by": "operator identifier",
  "provenance": {"source": "operator-reviewed-associations.json"},
  "samples": {
    "sample001": {"path": "approved/photo.jpeg", "sha256": "<64 lowercase hex>"}
  }
}
```

Each association contains exactly one path and SHA. Shared images are permitted;
there is no default image, session-based inference, or automatic quality selection.
Image **byte SHA** is verified, not photographic suitability or successful image
decoding. Fake image bytes in tests demonstrate only the validator contract.

Caption schema:

```json
{
  "schema": "h3-production-caption-mapping-v1",
  "approved": true,
  "approved_by": "operator identifier",
  "provenance": {"source": "approved-session-captions.json"},
  "scope": "session",
  "captions": {"session001": "Explicit operator-provided caption."}
}
```

`scope` is `session` or `sample`. Keys must cover exactly the frozen sessions or
715 sample IDs respectively. Values must be nonempty strings; text is preserved
verbatim. Both mapping files require approval, approver and nonempty provenance
with a `source` string. Additional provenance fields are retained. These declarations
are operator evidence, **not authenticated signatures**.

## CLI / API

```sh
python3 -m research.h3_personal.production_inputs \
  --selection /inputs/frozen_selection.json \
  --manifest /mounted/raw/manifest.json --summary /mounted/raw/summary.json \
  --source-map /inputs/mounted-source-map.json --source-root /inputs/sources \
  --media-root /mounted/raw --reference-root /mounted/references \
  --reference-mapping /inputs/approved-reference-mapping.json \
  --caption-mapping /inputs/approved-caption-mapping.json \
  --code-root /inputs --expected-input-revision <independently-pinned-40-hex> \
  --output /work/production-input-stage
```

Never derive the expected revision from the summary being checked. CLI output is
an exclusive new directory containing `rows.json` and `receipt.json`, written only
after all checks pass; existing stages are never overwritten. An I/O failure during
writing may leave a partial directory: require both files and verify the row hash
before consuming it. The Python API `stage(...)` returns `(rows, receipt)` with the
same keyword names (file arguments use the `_path` suffix).

Rows contain `id`, absolute resolved `video`, `audio`, `reference_image`, explicit
`caption`, `width`, `height`, `num_frames`, and unchanged `quality_pool`. Pass rows to
`component_encode.make_plan` separately with explicit encoder identities/attestor.
That planner rechecks/hashes its own inputs; this receipt does not bypass its checks.

Receipt retains both complete mapping documents, input file paths/byte hashes,
dataset/revision/code pins, row SHA and check scope. It explicitly reports
`authenticated_provenance=false`, `legacy_cache_certified=false`,
`technical_training_ready=false`, and `c_gate_enabled=false`. Hash consistency does
not authenticate historical producer execution or the revision's contents, prove
media decode/encoder numerical equivalence, certify legacy cache reuse, publish an
index, enable gate C, or authorize paid work. This stage does not run ffmpeg; actual
media decoding remains the producer/encoder's separate responsibility.

## Tests

```sh
python3 -m unittest discover -s tests -p test_h3_production_inputs.py -v
```

Fixtures create six tiny fake MOVs, 715 fake clip/audio pairs and a fake reference
inside temporary directories; no original recording/photo access is required.
Coverage includes planner compatibility/pool preservation, hashing deduplication,
completion/revision/freeze/code/dataset pins, ordering/source/range proofs, mounted
byte corruption, mapping approval/coverage, explicit captions, unsafe paths and
symlink escape, duplicate keys, and exclusive CLI staging.
