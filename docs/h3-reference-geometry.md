# Ref2VA reference geometry audit and local correction previews

## Confirmed bug, not yet proof of the entire output distortion

The pinned Diffusers revision `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
`resolve_reference_image_size(width, height)` returns **(height, width)**.
Our deployed `PinnedBackend.image()` assigned the result to `width, height`, then
passed it to `prepare_reference_image(original, height, width)` in reverse order.
This was verified both against the exact pinned source and the immutable runtime
used by the completed four-condition Job. It is not an inference from facial appearance.

For the actual reference SHA
`6789db42c765d2611a640bc9b47934b4912584e45c800d2700b1fddf9902faeb`:

| Stage | Width × height |
|---|---:|
| Source JPEG, EXIF orientation1 | 1737 × 3088 |
| Correct reference preparation | **2048 × 3648** |
| Legacy preparation actually deployed | **3648 × 2048** |
| Target video (raw clip and generated) | 768 × 1344 |

The legacy reference preparation introduces approximately **3.1667× horizontal
anisotropy** relative to its source. This is **NOT** the factor to invert on generated
video: the generator is not a simple resizer. Fixing reference preparation has not yet
been shown to eliminate all output deformation or preserve adapter identity.

Pinned packing source SHA:
`1f025b68af3f5c4316e2b89b026d4976c5b85286dafcc8b5c1d8ceb186c9bc01`.
Deployed component source SHA:
`2114f05051974013db7b538ff38dfaa654398109dbc94bbbbb4ed311511afd6c`.
The original reference is already correctly oriented (EXIF1), so EXIF rotation is not
this observed bug. The local raw clip has SAR1:1/DAR4:7; generated video is768×1344
without explicit non-square SAR. Target video loading uses aspect-preserving cover/crop.
Identical container sizes do not guarantee identical content proportions or player layout.

## Preserve legacy assets; corrected preparation is explicitly versioned

`research/h3_personal/reference_geometry.py` provides:

- `legacy-wh-v1`: exact historical behavior for reproducibility.
- `correct-hw-v2`: correct height/width assignment, no additional face manipulation.

`continuation_backend.CorrectedBackend.image(unit, geometry_policy='correct-hw-v2')`
exposes the corrected-only producer. `component_encode.PinnedBackend` remains byte-for-byte
legacy: its full source hash is part of all frozen cache producer identities, so even a
backward-compatible signature change invalidates reuse. The first native GPU attempt
failed at that gate; the producer was restored and all1380 identities revalidated on CPU.
No gate was relaxed. See `h3-native-continuation.md` for current execution status.
This compatibility path is known-bug reproduction, not the preferred new geometry.

**No715cache mutation, retraining, or adapter replacement was performed.**
A future corrected-reference inference must separately prepare and pin:

1. Corrected reference pixels and new reference VAE latent/geometry.
2. Matching text/image conditioning and presentation coordinates.
3. A new policy-bearing cache identity and receipt, isolated from legacy artifacts.
4. Same original64 adapter and fixed audio/prompt/seed/output geometry for A/B.

The existing cache orchestration has not yet been wired to produce/accept this new
policy end-to-end. Merely flipping the image helper in a production preview is unsafe.
Whether original64 has adapted to the old conditioning distortion is unverified; do
not discard it or declare retraining necessary without controlled evidence.

## Long recorded-audio validation: native15s limit

The user next requested corrected-reference Turbo+2GPU inference using the entire
`voice-30s.wav` (actual28.4231667s), with a supplied Japanese transcript. The pinned
Diffusers `packing.py` defines `MINIMAX_H3_MAX_DURATION = 15.0`; trainer Geometry
rejects longer generations. Do not silently remove this guard or treat arbitrary
CLI duration parsing as evidence of trained long-duration support.

The local proposal is two345-frame14.375s windows, at global frame0 and338 (24fps),
with7frames overlap, then assemble683frames/28.4583s. Mux the unchanged original
full audio once, continuously. H3 input still requires native32kHz stereo derivative
and window slicing; source WAV is not overwritten, denoised, stretched, or synthesized.
A window join can produce a visible pose/expression discontinuity; neither overlap
nor concatenation establishes causal/gapless continuity. No automatic face blend.

This is a pending proposal, not an implemented/GPU-verified long-audio path.
The old exact-rate exception covered the four-condition campaign only; a new scoped
HF h200x2 approval is required (proposal$10.00002/hr,<=4800s/$13.34). No paid launch,
new reservation or remote upload has occurred for this request. Local plan lives in
private `gpu-hf-corrected-recorded-30s-v1/`. Corrected reference conditioning/cache
identity integration must be finished and tested before deployment.

## CPU-only fixed-scale preview tool

```bash
PYTHONPATH=. python3 -m research.preview_h3_geometry \
  --input /private/generated/review.mp4 \
  --output /private/new-preview-directory \
  --ratio .75 --ratio .80 --ratio .85
# Plan-only by default. Add --execute to render local candidates.
```

`--ratio` is the desired **horizontal/vertical scale ratio**, fixed over the entire
clip. It is not a fitted face ratio and not a camera/FOV recovery model.

- **pad:** horizontal shrink, same height, centered black side padding. Preserves all
  source content but adds visible bars and reduces apparent subject size.
- **crop:** vertical stretch by approximately1/ratio, same width, centered vertical
  crop back to the original canvas. No bars, but loses approximately1-ratio of the
  vertically stretched field. It cannot reconstruct missing image content.

Both use Lanczos, square pixels, original canvas, unchanged frame count/duration,
and copied audio. Audio packet SHA sequence is independently verified, outputs
are fully FFmpeg-decoded, originals are hash-checked unchanged. Video is re-encoded
(H.264 CRF18); it is not pixel-lossless or identical to an uncorrected render.
Input limits:64MiB,15seconds, one bounded even square-pixel video, explicit rotation
must be audited first. New exclusive output directory; no overwrite, upload,
network, GPU, automatic playback, per-frame tracking, or generative face enhancement.

## Actual local candidates and automatic-estimation limit

Private folder:
`$H3_PRIVATE_ROOT/gpu-hf-four-condition-matrix-v1/geometry-audit-v1/`.

- `geometry-audit.json`: pinned-source reproduction, original/reference dimensions.
- `reference-legacy-reproduction.png`, `reference-corrected-v2.png`: **private
  reference-derived media**, never upload as generated-only output.
- `baseline-candidates/`: ratios.85/.90/.95, pad+crop,6generated preview videos.
- `turbo-candidates/`: ratios.75/.80/.85, pad+crop,6generated preview videos.
- `baseline-comparison.png`, `turbo-comparison.png`, `review.html`: local review UI,
  includes private reference pixels. No autoplay, external assets, or publication.
- Per-folder `preview-report.json`: hashes, actual integer-rounded scale, crop cost,
  preserved audio packets/frames/duration, quality and ratio remain unaccepted.

Local existing OpenCV SIFT/RANSAC background matching was attempted without installing
anything or using face landmarks. It produced only6normal inliers and5Turbo ratio
matches; neither was accepted as a reliable global scale estimate. Perspective/FOV
changes confound single-affine recovery. The preview ratios are **manual candidate
sweeps**, not automatic calibration or claimed ground truth. Do not turn the low-support
fit into an automatic production warp. Future automatic calibration needs confidence,
spatial coverage, perspective/pose discrimination and temporal stability gates.

Tests: `python3 -m unittest discover -s tests -v`:479passed/63skipped, including
portrait/landscape/square H/W regression and synthetic FFmpeg audio/frame preservation.
No additional GPU cost; existing compute ledger remains unchanged.
