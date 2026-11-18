# Shared component cache foundation (experimental)

Gate C remains **false**. This is executable CPU validation/index publication and
an exact-source dataset-loader patch, not encoder execution, training completion,
or authorization to train. No weights, downloads, uploads, or jobs are involved.

## Scope and use

The source-only installer first invokes the existing exact guidance source plan
(all Aviad SHA256 pins and patch contexts), then composes the shared-path loader:

```sh
python3 research/install_identity_component_cache.py /tmp/h3-identity-audit/aviad
python3 research/install_identity_component_cache.py /tmp/h3-identity-audit/aviad \
  --output /tmp/h3-component-trainer  # must not exist
```

A new source tree includes both runtimes. The original pinned source and existing
installer/runtime remain unchanged. Use the guidance training subset: batch 1,
one always-on image reference, real clean target audio, generated video,
Ref2VA LoRA. The existing guidance configuration restrictions still apply.
The patched `PrecomputedDataset` requires exactly these source keys for v2:
`video`, `audio`, `text`, `reference`, `identity_uncond`. Directory values are
ignored for v2; artifact locations come from validated component descriptors.
Both conditional and paired-unconditional loads actually follow shared paths. Sequence budgeting includes reference rows and the longer conditional/probe text branch; v2 over-limit samples fail rather than silently being dropped. The composed trainer initializes its previously missing skip counter.
V1 trees retain the original fixed-ID guidance paths and metadata validation.

## Catalog and fresh v2 index API

`research.h3_personal.component_cache` has no torch/safetensors imports until
validation/loading. Tensor operations require those dependencies in a separate
environment. Start with already-produced artifacts in a contained cache root:

```python
from research.h3_personal import component_cache as c

entry = {
    "kind": "video",
    "path": "objects/target.safetensors",  # relative, existing, contained file
    "sha256": c.sha256(cache_root / "objects/target.safetensors"),
    "spec": {"width": 32, "height": 32, "num_frames": 22},
    "attestation": {
        "aviad_revision": c.REVISION,
        "recipe": "exact recipe description from the producing run",
        "source_sha256": source_digest,
        "encoder_sha256": encoder_identity_digest,
        "attested_by": "responsible producer/run identifier",
    },
}
key = c.component_id(entry)
# Assemble all five kinds and complete samples before publishing:
# c.validate_catalog(cache_root, catalog)
# c.write_index(cache_root, catalog, samples)
```

The component address is SHA256 of canonical JSON of the whole descriptor
**except `path`**. It includes kind, spec, attestation and artifact hash. Moving
identical artifacts does not change their address. This module validates a
catalog; it does not create artifacts or implement a component execution planner.
Artifact names need not contain sample IDs. Multiple samples can reference the
same address. `write_index` validates every artifact and sample before opening a
new `index.json` exclusively. It never merges or overwrites an old index. This is
a single-writer foundation, not an atomic crash-recovery/locking protocol.

Descriptors have exactly `kind`, `path`, `sha256`, `spec`, `attestation`.
Supported specs and actual safetensors contracts:

| Kind | Spec | Tensor contract |
| --- | --- | --- |
| video | width, height, num_frames | `latents`: float32 `[derived_video_rows,96]` |
| audio | rows | `latents`: float32 `[rows,32]` |
| reference | geometry: `[0,1,Hlatent,Wlatent,0]` | `geometry`: int64 `[1,5]`; `latents`: float32 `[Hlatent/2 * Wlatent/2,96]` |
| text | rows | `prompt_embeds`: bfloat16 `[rows,5120]`; `text_token_tags`: int64 `[rows]` |
| identity_uncond | rows, conditional, reference | same text contract; dependency values are component addresses |

No extra tensors are allowed. All float tensors must be finite. Validation is sequential and retains only small vision-row summaries, not the full multi-GB tensor corpus. Sample IDs must be safe and unique. Reference latent
spatial axes must be positive and even; image latent time is exactly 1; reference
audio is forbidden. Tags are only video=0/text=1, with both present. Conditional
and probe vision-tag counts must match. Target dimensions must be positive
multiples of 32, frame counts `17*n+5`, 22–360. Derived video time is `5*n+2`;
audio rows are `2*round(frames/24*40)` (stereo channel-major). Short geometry is
allowed for synthetic smoke tests, not evidence of useful training distribution.

Each sample contains pinned `SampleRecord` fields (`id`, dimensions, frames,
video/audio/text row counts, `has_audio: true`, optional caption),
`reference_geometry` matching the actual reference tensor, and `components`
mapping **all five** kinds to addresses. IDs must be unique. Row counts must
match actual artifacts and geometry. All catalog entries must be used; empty or
incomplete samples fail rather than silently skipping. No placeholder audio,
keyframes, reference video/audio, or batch >1 support.

### Shared probe provenance, not fixed-ID metadata

A shared probe safetensors file must have exactly this string metadata:

```python
{
    "schema": c.PROBE_SCHEMA,  # h3-component-probe-v2
    "aviad_revision": c.REVISION,
    "probe_prompt": " ",
    "conditional_sha256": catalog[conditional_key]["sha256"],
    "reference_sha256": catalog[reference_key]["sha256"],
}
```

No `sample_id`: deduplication is bound to identical conditional/reference bytes,
not to one selected sample's name. Catalog dependency addresses must have the
correct kinds, and each sample's paired probe must point to that sample's exact
conditional and reference components. The composed runtime branches to this
validated v2 contract before guidance's fixed-ID path construction; it does not
weaken v1 metadata validation. Initialization validates the entire catalog before
split/token filtering; subsequent reads recheck path containment and file hashes.
Concurrent mutable artifact writers are unsupported (hash/load is not a lock).

### Legacy provenance stays unresolved

`provenance_status` returns `unresolved` without a structurally valid explicit
attestation; no index is published or loaded from such components. Old artifacts'
shape, bytes, directory names, sample IDs or v1 guidance metadata alone do not
prove encoder recipe, source identity or encoder identity. Attestations are caller
assertions with the exact revision, nonempty recipe/attestor, and SHA256 source
and encoder identities. They are **not** signed evidence or independent
verification of encoder execution; fabricated/synthetic claims cannot open C.
A producer must supply meaningful identities (including relevant preprocessing,
reference presentation and encoder configuration) before real reuse is justified.
There is no automatic legacy migration or invented provenance.

## Tests and remaining work

```sh
python3 -m unittest discover -s tests -p test_h3_component_cache.py -v
/tmp/h3-media-env/bin/python -m unittest discover -s tests -p test_h3_component_cache.py -v
```

Root: 3 static tests pass, 5 tensor tests skip when torch/safetensors are absent.
Separate environment: all 8 tests pass with synthetic CPU tensors. Integration
installs the real pinned source and instantiates its `PrecomputedDataset`, loads
two samples sharing all five artifacts, tests batch rejection and changed bytes,
and loads a v1 fixed-ID guidance cache through the same patched source. No model
is loaded. The local pinned fixture is required for installer/integration tests;
its absence is an explicit skip. Validation negatives cover shape/dtype/NaN,
geometry/tags, containment, hashes, unresolved attestations, probe metadata and
pairing, completeness, and duplicate IDs.

Remaining: actual component-wise encoder execution and resumable reuse, shard
scheduling/merge, recipe identity generation, signed/trusted provenance, atomic
publication/recovery, scale/performance work, GPU/model tests, real-media encoder
validation, adapter loading, and training/acceptance evidence. Existing pinned
preprocessing still writes v1; it is not converted into a component orchestrator.
