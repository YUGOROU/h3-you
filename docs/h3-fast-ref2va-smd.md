# Four-forward Ref2VA + personal identity candidate

## Objective and current gate

USER requires **both personal identity and realtime**, not a slow native Ref2VA
fallback. Native50-forward code is diagnostic only. This implements a separate
Ref2VA-trained four-forward candidate with conditioning reuse and bounded encoding.

Current budget authority/reconciliation: `h3-compute-budget.md` (HF additional$6,
Modal$20.44 shared with live inference). Old$0.65/$170 stop points below describe
v3 history and are superseded. Profile and compile follow-up:
`h3-ref2va-runtime-adoption.md` (DiT dominates; compile0.712× still FAIL).

**GPU validation completed: reference/adapter/cache/496-frame export gates passed;
realtime gate FAILED (0.69327×).** Human identity review, numerical parity,
motion continuity and client playback remain unverified. This is
not a claim that realtime personalization is achieved. No new raw uploads, large
weight download to the Mac, source checkpoint mutation or dependency installation.

## Selection and immutable evidence

Selected `zjwfufu/SMD-MiniMax-H3-Ref2VA-4step`
@`8f67bf6b8b017bf16be4fd36fdfe135eeb78715a`:

- Explicitly Ref2VA-trained, four forwards, image references are appearance
  conditioning rather than first-frame anchors. See pinned model README/sampling.
- `lora/adapter_model.safetensors`,1383677792bytes, SHA
  `b8cd339e167ab29e112d4f8934d3ac7f20b7e3269895fb149761bd959336bdd7`.
- Actual safetensors header:624BF16tensors/312pairs, rank128/alpha128, no replacement
  tensors. All208original identity targets match names/input/output dimensions.
- Trained base `MiniMaxAI/MiniMax-H3` @`939557dc319dd91227e30195a763f272ba7f8765`.
  All **61runtime-component files** are byte-equivalent by pinned Hub metadata
  (payload SHA for LFS, Git blob for ordinary files) to existing base @`42ed227ee7df40d41602854ae760620d6eb651fe`.
  Includes native14-shard transformer_ref, both VAEs/schedulers, Qwen/processor/tokenizer
  and model indexes. Not an assumption that two revision strings are interchangeable.
- Original64 remains SHA `43fd1f8c6e4097611ba027dcf42503b4c889cfd5ef5e45d91abcb08c482ef151`,
  rank16/alpha16. Derived constructor adapter concatenates overlapping rank axes:
  208pairs rank144, remaining104 rank128, alpha/rank1. Sources unchanged; finite and
  full serialization reload checks before GPU constructor. Actual full-size derived
  artifact is **not yet built**; serialization tests use full-count small fixtures.
- Private audit receipt:
  `$H3_PRIVATE_ROOT/fast-ref2va-smd4-v1/feasibility.json`.
  Auditor fetches metadata/header only, verifies cached original64 full SHA; no GPU.

Rejected as evidence of a Ref2VA-trained fast model:

- FastVideo8-Step V2 card says T2VA and untested reference modes.
- `maxsssssss/MiniMax-H3-PDD-LoRA` @`cbf70a06f14fe0a755203108251b6127791a330f`
  explicitly trains **t2va layout**. Presence of Ref2VA PDD classes in FastVideo does
  not establish an available Ref2VA-trained PDD checkpoint. Earlier proposed PDD
  direction was a source lead, not verified compatibility.
- No indiscriminate T2VA FastH3 gate/dense-payload stacking onto the SMD adapter.

## Runtime implementation

`research/h3_personal/fast_ref2va.py`, opt-in in `research/run_h3_ref2va.py`:

- Native Ref2VA class/transformer_ref/ordered own image, not FL2VA `pil_image`.
- Actual **four-forward SMD sampler**: raw final segment scale0.5, grid
  `[1,5/7,3/7,1/7,0]` (published iterative Python floating-point values), then shift
  once per modality12/3. Not uniform5points or integer-rounded DMD rungs.
  Exact Python results independently compared with functions extracted from the
  pinned published runner. FastVideo explicit sigmas are already shifted.
- Both ranks must attest actual Ref2VA,312total consumed LoRA pairs,208rank144
  personal overlaps,104rank128 pairs; zero unmatched weights. No missing-key warning
  treated as success. GPU merge numerical parity with published PEFT fusion is
  unverified; concatenation algebra alone does not prove rounding/quality parity.
- Four genuinely generated requests, seed314159..314162, each124frames256x448/24fps.
  Original reference remains present on every request; no returned-frame replay.
- Content-bound one-slot warm conditioning cache bypasses **Qwen move/reload and
  DiT parking**, not merely Qwen computation. Key includes reference file SHA,
  prompt and geometry/fps/duration; cache miss executes ordinary conditioning and
  replaces the slot. Only encoder output/tags cached; noise and RF augmentation
  remain freshly computed by latent stage. Both ranks must record three real hits.
- Pinned content-keyed VAE reference memo enabled with capacity1. Model/adapter/source
  lifetime is fixed to the constructed instance. No persistent private latent upload.
- Explicit sequential cold loading avoids co-resident Qwen+DiT peaks. LoRA setup
  is deferred until DiT materializes, then guarded before any denoise; the driver
  collects both receipts after the cold request and before accepting/encoding it.
  Warm cache hits bypass the sequential park/reload path. No unadapted warm run.
- SP2/TP1, parallel VAE gather, compile off; bounded one-active/one-waiting CPU encoder
  overlaps later GPU requests. Existing producer keeps its default FL2VA timing;
  new explicit timing factory emits all124frames of each Ref2VA request.
- **496frames**, not493: reference images are not shared generated first frames,
  so trimming a supposed shared head would be wrong. These are independent clips;
  identity consistency and seamless motion continuity require separate review.
- Independent warm generation, consumer and local-availability throughput gates,
  plus no availability deadline miss. No initial-buffer manipulation can compensate
  for generation/consumer throughput<1. Local encoded availability is **not** private
  Hub publication, Mac playback, network latency or causal AV evidence.
- Generated audio discarded, visual-only. No Irodori conditioning/audio sync claim.

### Locked external speech

`--locked-speech <wav> --locked-speech-sha256 <sha>` (SMD only, resident
visual-only policy) locks a supplied utterance onto every denoising forward for
lip-sync instead of only generating discarded audio. Implemented and CPU-audited,
no GPU run; see `docs/h3-smd4-locked-audio.md`.

### Important remaining performance constraint

SMD has **no trained H3 VSA gates**. VIDEO_SPARSE_ATTN_H3 therefore stays disabled;
no zero/random gates and no silent T2VA gate fusion. Dense FLASH_ATTN is explicit.
Four forwards and warm encoder bypass may improve nativeRef speed substantially,
but **do not justify a realtime prediction**. If the measured generation lane is
below1x, the implementation is not accepted: need compatible reference-preserving
sparsity/distillation/pruning, not removal of the reference or identity adapter.

## Usage / next allocation

Use the same full local native snapshot, original64, own reference hash and verified
source archive described in `h3-fastvideo-ref2va.md`, adding:

```sh
# Default remains full offline preflight; no GPU inference/submission.
python -m research.run_h3_ref2va [existing required arguments] \
  --fast-ref-smd --smd /private/smd-ref2va4/lora/adapter_model.safetensors
```

Only explicit `--run` on the pinned provisioned CUDA runtime executes. Saves four
local fMP4 segments and profile, never auto-publishes or autoplays. A new paid Job
needs its own immutable deployment, source overlays, fresh private-input/pin/
permission/license/idleness/rate checks and finite shared-ledger reservation.
Pre-submission bound165.80USD incl1USDmargin; allocation below is now reserved.

### GPU validation authorized and submitted

USER `gpu検証までgo`, then `continue, go`: corrected Job
**`YUGOROU/6ad231e6c66288145225d208`**, private dataset
`YUGOROU/h3-fast-ref2va-smd4-gpu-20261016-v3`
@`a9c83739d89e47834446dcb830b11f0f7aafa005`.
Local `gpu-hf-fast-ref2va-smd4-v3/`; **never replay submitted launcher**.

Prior Job `6ad1ee245ccb47ba0f7b1c24` CANCELED while SCHEDULING, both
started_at/finished_at absent. Reservation reconciled to0 runtime bound (not an
invoice claim;1USD ancillary margin retained). Diagnosed before allocation:
output inside deployed repository would fail the runner's private-output guard.
v2 uses sibling `/work/fast-ref2va-generated`, covered by a regression test;
new private repo/commit/guard, no old-deployment replay.

v2 `6ad22e68c66288145225d05e` ERROR, started14:02:20.794Z,
finished14:14:21.821Z on2026-10-16,721.027s/2.01USDupper runtime bound.
Both ranks loaded208personal/312total with0unmatched and cold124-frame generation
returned. First warm request failed in packing.py484: packed metadata is CPU,
but cache restoration wrongly moved text_token_tags to GPU. v3 retains tags on
CPU and embeddings on rank-local GPU, with a meta-device regression that
reproduces the wrong placement without CUDA. Full589tests/88skipped and8Torch
candidate tests passed. No generated outputs published from failedv2, so no
identity/RT claim. Its reservation was replaced by2.01USD runtime bound.

- One finite780s/2.166671USDmaximum Job, h200x2 rate10.00002/hour, no campaign.
- Reservation was169.976671USD including1USDmargin; terminal reconciliation
  is now169.35USD,0.65USDunreserved. Runtime bounds/reservations are not invoices.
- Live all-HF-terminal/all-Modal-environments0tasks, PRO/permissions, hardware rate,
  private immutable original64/reference and all67selected input-file metadata
  checked before submit. Existing HF operator acceptance preserved; unknown
  execution country is not represented as verified permissible territory.
- `prepare_h3_fast_ref2va_gpu.py`, `launch_h3_fast_ref2va.py`,
  `bootstrap_h3_fast_ref2va.py`. Shared launcher retains original defaults while
  exposing an explicit inspector/bootstrap/USER-request hook for this separate scope.
- Only download/hash IO now uses bounded4-thread parallelism for base assets;
  inference and timing gates unchanged. Timeout reduced to fit remaining2.19USD.
- GPU downloads selected native14shard base/shared assets, pinned SMD and private
  original adapter/reference; no new raw upload. Shared real FFmpeg probe before
  weights, new Ref124-frame PTS+full-decode CPU test passed. No package installs.
- Both-rank coverage and warm-cache receipts mandatory; fixed four generated clips.
  Only four MP4s and profile uploaded privately after success and full decode;
  derived weights, input images and cached tensors never published.
- Job COMPLETED, but **server realtime gate FAILED**. Identity requires USER review.

### Verified GPU result (v3)

2026-10-16T14:17:14.867Z→14:26:29.183Z,554.316s/1.54USD runtime upper bound.
Outputs immutable commit `add68bdcae3b14d45a9f0c7410c989e7e3e27eac` in the
private v3 dataset. Four generated files + profile retrieved with remote content
hashes checked; every segment full-decoded and actual contiguous PTS verified.
Offline lossless joined `verified-results/review.mp4`:496frames/20.666667s,
SHA256 `f6e76fe66de1655b2b104e8a5395f505a951a840ca1c400b8a98577dacb61c8d`.
`contact-sheet.png` available privately; nothing autoplayed.

- Both ranks:208identity/312total/0unmatched, exact rank144/128 composition.
  Full serialized reload passed, derived adapter SHA256
  `b4bb246ab2bce2080e316905b61d55cb231fcb712a0f26cdde21b8f8f563fe8f`.
- Both rank caches:3warm hits. Original reference retained, no T2VA substitution,
  no previous-frame reference, no shared-head trim. Independent clips, not
  demonstrated seamless motion continuity.
- Warm generation0.69397833×, **local encoded availability0.69326955×**;
  consumer/encode24.281127×. Each5.166667s clip takes7.43–7.47s generation.
- Lateness0/2.281400/4.550222/6.857826s: genuine growing backlog, strict FAIL.
- Cold first local encoded availability195.921627s runner scope, excluding
  bootstrap/downloads; not an end-to-end cold-start guarantee.
- USER review: mostly resembles USER in face, outline and build; especially later
  degradation and slight head jitter remain. USER considers consistency promising
  for this speed. Record as **partial identity acceptance, not full identity pass**.
  No lip-sync or causal external audio.
- `profile.json uploaded=false` describes the isolated no-Hub runner. Bootstrap
  subsequently published generated-only outputs privately; remote commit verified.
- Need approximately1.44× higher generation throughput (~31%lower latency) while
  retaining identity+reference. No new GPU allocation: remaining0.65USD cannot
  cover the observed fresh cold-load path. Do not revert to50steps or reference-free
  T2VA and do not silently stack untrained VSA gates.

### Post-review CPU helpers (no new paid allocation)

Reuse registry search for temporal-reference audit/stage timing found no published
match. Reused the project's native-video FFmpeg/Pillow sheet approach; no new
packages or device changes. Frozen v3 files, original references and adapters stay
unchanged.

- `audit_h3_ref2va_review.py`: SHA-bound496-frame review sheets. Each of four clips
  shows local frames0/31/62/93/123; separate sheets show122/123/124/125 around
  independent-clip boundaries. This separates possible within-clip deterioration
  from seed/pose/framing changes at cuts; it does **not** automatically diagnose
  the user's reported later degradation or estimate identity/jitter scores.
  Generated locally at `verified-results/drift-review/`, no upload/autoplay.
- `ref2va_helpers.aspect_candidate`: same-pixel-area geometry planner only.
  Actual reference1737×3088 is9:16, current video448×256 is7:4. Aspect mismatch
  factor3.111 versus1.01587 for a256×448 portrait candidate. Matching orientation
  may help framing/proportions without reducing reference resolution; this is a
  hypothesis, not demonstrated quality/speed improvement. Equal pixel area does
  not guarantee equal latency. No automatic shape change or source-image warp.
- Optional `--profile-warm-stages` on the isolated native runner in explicit
  `--fast-ref-smd` mode records six per-rank/per-hit reports: latent preparation,
  denoising, video decoding and discarded-audio decoding. Host-call duration and
  CUDA default-stream elapsed intervals are separate; do not sum ranks or call
  these summed kernel times. One added synchronization per warm request makes
  this **diagnostic, not a comparable realtime benchmark**. Off by default;
  profile records `benchmark_comparable=false` when enabled. No guessed bottleneck
  or claimed speedup before GPU measurements. Deployment preparation includes
  the new helper module, but no new deployment has been submitted.

Next: review sheets with USER to locate drift; profile compute vs VAE vs metadata
before optimizing. Keep own-reference SHA/LoRA coverage/cache gates. Investigate
same-area portrait geometry as a separately validated identity-maintenance
candidate; do not simply lower reference resolution, repeat seeds to fake
continuity, substitute T2VA, or activate incompatible sparse gates.
Tests:594 passed/88 skipped with stdlib environment; pinned-source Torch candidate
suite8/8 also passed after warm-stage wiring. CUDA profiler itself and portrait
candidate remain GPU-unverified. Remaining budget0.65USD unchanged.

## Tests

- Dedicated CPU8/8 with Torch+safetensors and pinned archive: full-count derived
  serialization, source overlay AST, source invariance, header rejection, base
  payload comparison, no lost Ref frames, anti-prebuffer gate, warm cache skips
  encoder while still executing latent/denoise/decode stages and invalidates on
  changed image bytes.
- Full suite589OK/88skipped; GPU deployment/encoder-specific4/4 passed. GPU and client gates remain open.
