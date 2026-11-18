#!/usr/bin/env python3
"""Stdlib-only exact-source installer. Creates a NEW source-only trainer tree.

No weights, network, jobs, in-place edits, monkeypatches, or gate-C enablement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import tempfile

REVISION = "7a1563a267a17da898df3396acbd6a9d24fcb142"
# Entire src/h3_trainer Python tree, plus the actual preprocessing entry point.
PINS = {
    "src/h3_trainer/__init__.py": "03229a8161e772812fc42a8ef8593ec58d8c4a514b0385726a64b7b9ad3f911f",
    "src/h3_trainer/checkpointing.py": "77b0f29f4fc34b53ce4299eca33bd66c09846ba7e321e09ccbb0c7e8321e3da5",
    "src/h3_trainer/config.py": "23844c0f470b420b55f411ab90005802a75d5655c8d656857c5e2a2a4509e0f8",
    "src/h3_trainer/constants.py": "8c7e2d4d876edaca1d6e5f04ac17c82d85a2efadec2a892370fcaf3332f60f8c",
    "src/h3_trainer/datasets.py": "4e0b30d653f4cb7bd4f7a9883a4ed5f3aa56cbb8e2bfe60d37572f9e87127d79",
    "src/h3_trainer/flow_matching.py": "940047c1964859449f8656127c996aa8330e89edd20f238fcd1b58261390c412",
    "src/h3_trainer/inference.py": "b06ec584b43e3e7567bbc06b4616ea82870c85289b1a6002c3f67efcdc266d94",
    "src/h3_trainer/logging_utils.py": "e3c125cbde043f0f2b8aba7adf0a97f3beda7cfa68bf71bc0f419e534903ec6a",
    "src/h3_trainer/lora.py": "076e2bb3b0ae920bab49f4c56052403c5ab3ffc1ec7edf2932b9dc7d7f0b52df",
    "src/h3_trainer/model_loader.py": "1ee2590bb47ccaad70a8578537c1da9f311adb05ce594ef650004252e7569bf1",
    "src/h3_trainer/packing.py": "9e9b76b573ae54ee9a3d52ed09c7dc5539e9c2b8bd4e0506da0fa317bdd0ad6a",
    "src/h3_trainer/preprocessing/__init__.py": "04b22bd9620c5a0b9d4dcd4d69447685fb15411e4f144d5240b0026d31f6483f",
    "src/h3_trainer/preprocessing/builder.py": "2fd97ab8bc2f3876c805a2ecc3118637a5af2c018e0e62721a650540be85bd82",
    "src/h3_trainer/preprocessing/encoders.py": "b6d2689316c13165995895a11d231a0246594bb0093d8bdcde88cd5ba361c563",
    "src/h3_trainer/preprocessing/media.py": "c2923e41bd2fb056fb8e448ee24f465b95eca495497affb3ed326f0a6784072a",
    "src/h3_trainer/quantization.py": "62affd5b72742c2d8e7ff60e2a32767516d7601f2a05a8659442dce23d298f97",
    "src/h3_trainer/timestep_samplers.py": "da0026be7a4ac5dd33e62f649ccf80c456ab547c51b947d2fb677abceb113076",
    "src/h3_trainer/trainer.py": "681ebebdd24d71d499d5403e6695797d8afa7c446ea4548420defc0300323a5d",
    "src/h3_trainer/training_strategies/__init__.py": "0a53d3ee5ee16b2b6902f1f8371b793c0e01efecd891975f186378c78ea51efb",
    "src/h3_trainer/training_strategies/base_strategy.py": "c8766348d51b5655d52d600633cf14edabf61c2b59fc30eee2b3df9742321a0c",
    "src/h3_trainer/training_strategies/flexible.py": "6648a791d6dca04b9b9921d59a0d7f8f9c145af8147632bbe696959936e3afb1",
    "src/h3_trainer/validation_runner.py": "935db9676283877a82574d09e31ee3ddd8c3f30aa9d72e3dcef7c6d6a7190cfe",
    "scripts/process_dataset.py": "4fd9395de6a0f5b56cab4bc404f6a1f31efe81d9cf8fef20b1d33ac2e81f81fa",
}


def substitute(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Patch context mismatch ({count} expected): {old[:100]!r}")
    return text.replace(old, new)


def patch_sources(sources):
    result = dict(sources)
    path = "src/h3_trainer/config.py"
    text = result[path]
    text = substitute(text, '    name: Literal["flexible"] = "flexible"',
        '    identity_guidance_scale: float | None = Field(default=None, ge=1.0)\n'
        '    identity_guidance_sigma_min: float = Field(default=0.15, ge=0.0, le=1.0)\n\n'
        '    name: Literal["flexible"] = "flexible"')
    text = substitute(text, '        for modality in (self.video, self.audio):\n            for condition in modality.conditions:',
        '        if self.identity_guidance_scale is not None:\n'
        '            sources["identity_uncond"] = "identity_uncond"\n'
        '        for modality in (self.video, self.audio):\n            for condition in modality.conditions:')
    result[path] = text

    path = "src/h3_trainer/datasets.py"
    text = substitute(result[path], '        # Vision blocks embedded in the prompt',
        '        if "identity_uncond" in self.data_sources:\n'
        '            from h3_trainer.identity_guidance import load_uncond_cache\n'
        '            item.update(load_uncond_cache(self, record, text_payload))\n'
        '        # Vision blocks embedded in the prompt')
    # Guidance supports only batch 1; reject before the pinned collator stacks bool references.
    text = substitute(text, '    keys = {item["record"].bucket_key for item in batch}',
        '    if any("identity_uncond_text" in item for item in batch):\n'
        '        raise ValueError("identity guidance supports batch_size 1 only")\n'
        '    keys = {item["record"].bucket_key for item in batch}')
    result[path] = text

    path = "src/h3_trainer/training_strategies/flexible.py"
    text = substitute(result[path], 'from h3_trainer import logger',
        'from h3_trainer import logger\nfrom h3_trainer.identity_guidance import attach_probe, video_loss')
    text = substitute(text, '        return assemble(', '        packed = assemble(')
    text = substitute(text, '    def _collect_keyframes(',
        '        return attach_probe(packed, batch, sigmas, self.config, references, geometry, device)\n\n'
        '    def _collect_keyframes(')
    text = substitute(text,
        '            video_loss = flow_matching_loss(\n'
        '                prediction_video,\n                packed.clean_video,\n'
        '                packed.noise_video,\n                row_mask=packed.video_loss_mask,\n            )',
        '            video_loss = guidance_video_loss(prediction_video, packed, self.config)')
    # Avoid shadowing the imported function with the local video_loss variable.
    text = text.replace('attach_probe, video_loss', 'attach_probe, video_loss as guidance_video_loss')
    text = substitute(text, '            extras={"seq_len": float(packed.sequence_length)},',
        '            extras={"seq_len": float(packed.sequence_length),\n'
        '                    "guidance/applied": float(packed.metadata.get("identity_guidance_applied", 0.0)),\n'
        '                    "guidance/base_sigma": float(packed.metadata.get("identity_base_sigma", 0.0))},')
    result[path] = text

    path = "src/h3_trainer/trainer.py"
    text = substitute(result[path], 'from h3_trainer import logger',
        'from h3_trainer import logger\nfrom h3_trainer.identity_guidance import training_forward, validate_training_config')
    text = substitute(text, '    def setup(self) -> None:\n        config = self.config',
        '    def setup(self) -> None:\n        config = self.config\n        validate_training_config(config)')
    text = substitute(text, 'prediction_video, prediction_audio = self._forward(packed)',
        'prediction_video, prediction_audio = training_forward(\n'
        '                    self._forward, packed, self.config.training_strategy)', count=2)
    result[path] = text

    path = "src/h3_trainer/preprocessing/builder.py"
    text = substitute(result[path], 'from h3_trainer import logger',
        'from h3_trainer import logger\nfrom h3_trainer.identity_guidance import encode_pair, write_uncond_cache')
    text = substitute(text, '    encode_audio: bool = True',
        '    identity_guidance_uncond: bool = False\n    encode_audio: bool = True')
    text = substitute(text, '            if path.exists() and not options.overwrite and record.get("text_rows"):',
        '            if (path.exists() and not options.overwrite and record.get("text_rows")\n'
        '                    and not options.identity_guidance_uncond):')
    text = substitute(text, '                embeds, tags = self._encode(record, row)',
        '                if options.identity_guidance_uncond:\n'
        '                    embeds, tags, uncond, uncond_tags = encode_pair(self, record, row)\n'
        '                else:\n                    embeds, tags = self._encode(record, row)')
    # The new explicit mode must not silently skip an incompatible sample.
    text = substitute(text, '            except Exception as exc:\n                logger.error("[%d/%d] %s caption failed:',
        '            except Exception as exc:\n'
        '                if options.identity_guidance_uncond:\n                    raise\n'
        '                logger.error("[%d/%d] %s caption failed:')
    text = substitute(text, '            record["text_rows"] = int(embeds.shape[0])',
        '            if options.identity_guidance_uncond:\n'
        '                write_uncond_cache(options.output, sample_id, path, uncond, uncond_tags)\n'
        '            record["text_rows"] = int(embeds.shape[0])')
    result[path] = text

    path = "scripts/process_dataset.py"
    text = substitute(result[path], '    parser.add_argument("--references",',
        '    parser.add_argument("--identity-guidance-uncond", action="store_true",\n'
        '                        help="cache paired reference-bearing single-space text probe (experimental)")\n'
        '    parser.add_argument("--references",')
    text = substitute(text, '        encode_audio=not args.skip_audio,',
        '        identity_guidance_uncond=args.identity_guidance_uncond,\n'
        '        encode_audio=not args.skip_audio,')
    result[path] = text
    for path, source in result.items():
        ast.parse(source, filename=path)
    return result


def plan(source_root):
    root = Path(source_root)
    expected_src = {path for path in PINS if path.startswith("src/")}
    actual_src = {str(path.relative_to(root)) for path in (root / "src").rglob("*.py")}
    if actual_src != expected_src:
        raise ValueError("Pinned src Python file set mismatch")
    sources = {}
    for path, digest in PINS.items():
        file = root / path
        if file.is_symlink():
            raise ValueError(f"Refusing symlink source: {path}")
        payload = file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Pinned SHA256 mismatch: {path} (expected Aviad {REVISION})")
        sources[path] = payload.decode("utf-8")
    return patch_sources(sources)


def install(source_root, destination):
    """Validate every pin/context/AST before writing; atomically publish a new tree."""
    sources = plan(source_root)
    runtime = Path(__file__).parent / "h3_personal" / "identity_guidance.py"
    runtime_text = runtime.read_text()
    ast.parse(runtime_text)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"Destination must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".h3-guidance-", dir=destination.parent))
    try:
        # Copy ONLY verified Python source. No artifacts, weights, signing or caches.
        for path, text in sources.items():
            output = staging / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        (staging / "src/h3_trainer/identity_guidance.py").write_text(runtime_text)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="new source-only experimental tree")
    args = parser.parse_args(argv)
    if args.output is None:
        plan(args.source)
        print(f"Exact source/context verified: {REVISION}; gate C disabled")
    else:
        print(f"Installed source subset at {install(args.source, args.output)}; gate C disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
