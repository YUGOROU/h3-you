# Independently pinned H3 model asset preparation

`research/prepare_identity_model_lock.py` prepares byte expectations for
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.
It does not download model weights, install dependencies, upload artifacts, submit
Jobs, or assert compute-country/license approval. Python stdlib only.

## Run

```sh
python3 -m research.prepare_identity_model_lock \
  --output $H3_PRIVATE_ROOT/model-assets \
  --token-file ~/.cache/huggingface/token
python3 -m unittest discover -s tests -p test_h3_identity_model_lock.py -v
```

`--token-file` explicitly overrides `HF_TOKEN` without displaying either.
Without that option, `HF_TOKEN` is optional (public read access works where allowed).
Do not put tokens in command arguments or source. The output directory must be
private (0700); new files are 0600. Existing artifacts are never overwritten;
use a new private directory for another preparation.

## Trust and bounded IO

- Fetch the revision-specific model-info API with `blobs=true`, preserving its
  exact JSON response as `source-metadata.json`. Require the repository ID and
  returned commit to match the hardcoded independent pins, not `main`.
- Inventory all files under `vae`, `audio_vae`, `text_encoder`, `tokenizer`,
  `processor`, plus `model_index.json`. Inventory `transformer_ref` separately.
  Ignore unrelated top-level assets, docs, README/LICENSE and hidden cache paths.
  Reject unsafe/duplicate paths, absent components/configs/weights, missing
  Git blob IDs, malformed LFS digests and inconsistent sizes.
- LFS SHA256 and size come directly from this revision's API metadata. **No LFS
  bytes are fetched**, even small LFS files. This is an expected-byte lock, not
  evidence that weights were locally hashed or successfully loaded.
- For ordinary Git files, allow only JSON/text config/tokenizer assets. Fetch
  revision-pinned bytes, verify exact length and Git SHA1 over
  `blob <size>\0<bytes>` against the API `blobId`, then compute their exact SHA256.
  The large tokenizer JSON is permitted, not replaced by an alternative digest.
  Deduplicate identical Git objects across directories; enforce a **32 MiB total
  unique small-file download cap** before any file requests. Each response is
  bounded to its expected size (+1 byte for overflow detection), with a 60s
  request timeout. No partial or blob-only substitute lock on budget excess.
- Small bytes are transient in memory, not stored as model files. Validate
  weight-index references against the selected LFS inventory. Save artifacts
  only after all metadata and small-file checks succeed. The response metadata
  itself is separately capped at 8 MiB. Redirects must remain HTTPS and cannot
  forward authorization to another host; request failures omit response details.

The Hub HTTPS revision API is the metadata trust source; this is not a locally
reconstructed Git commit/tree signature proof or an encoding provenance receipt.

## Worker wiring

`model-lock.json` uses the existing exact schema:

```json
{"schema":"h3-local-model-lock-v1","revision":"42ed227ee7df40d41602854ae760620d6eb651fe","files":{"model_index.json":"<SHA256>","vae/config.json":"<SHA256>"}}
```

The illustrative `files` map above is abbreviated; the real lock covers every
relevant component file. Supply its path and **byte SHA256** as the pipeline's
`model_lock`. This matches `hf_identity_pipeline.validate/provision` and
`hf_identity_encode.verify_models` without changing either worker.

`transformer-assets.json` contains a separate `files` SHA256 map and sorted
`transformer_files` list. Copy that list into the pipeline request, keeping
transformer assets out of the component lock so the encoder's exact inventory
check still works. The existing provisioner authenticates actual transformer
bytes against revision-specific API LFS/Git metadata; it does **not** consume
this separate transformer's SHA256 map automatically. Keep its artifact pin for
review/deployment rather than claim additional enforcement already exists.

`source-receipt.json` records each expected file's size, Git blobId, exact SHA256,
LFS metadata where applicable, and source method. It also pins the byte hashes of
the lock, transformer inventory and raw metadata response. Pin the receipt itself
externally if transferring it. No credentials are recorded. Preparation does not
relax the provisioner's independent runtime, private-repository, territory,
license, budget or byte-verification gates. No actual H3 runtime acceptance has
been established by these tests or metadata preparation.

## Actual preparation result

Successfully prepared outside the repository in
`$H3_PRIVATE_ROOT/model-assets/`:

| Artifact | Byte SHA256 |
| --- | --- |
| `model-lock.json` | `f1bb6420d2ad2406d88b5fb6dd777bad89808274a185335cd72a90fa380d8e9d` |
| `transformer-assets.json` | `dcbf82442da76d30fed693a001cd662774a47a5e4bdf0a5974bf34fd8cc36ba6` |
| `source-metadata.json` | `93e373dc68f72071551265ddd5296be48d9e8dbc789431afae00c069ec28acca` |
| `source-receipt.json` | `d5aedf1a67384c2335eb30915240151ecff2b35f751fb2111cafc9b20ff6e9ca` |

- 42 component files; 16 separate transformer_ref files.
- Component LFS/weight bytes expected: **77,735,901,100**.
- Transformer_ref LFS/weight bytes expected: **66,280,504,216**.
- Total expected weights: **144,016,405,316 bytes**; downloaded weights: **0**.
- Actual unique small-file bytes downloaded: **11,744,137**. Including repeated
  tokenizer/config paths, logical small-file inventory is **34,734,567 bytes**;
  deduplication keeps actual download/memory below the 32 MiB cap.
- Raw API JSON: 54,304 bytes (additional to small-file downloads).
- Seven fake API/IO tests pass, including exact compatibility with the existing
  encoder using synthetic local assets. No Jobs or compute-country claims.
