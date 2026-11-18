#!/usr/bin/env python3
"""Compose strict canonical export into the exact pinned sampling trainer tree."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import tempfile

try:
    from research import install_identity_sampling as sampling
except ModuleNotFoundError:
    import install_identity_sampling as sampling

# This wrapper supplies reviewed provenance labels, not authenticated base evidence.
PIPELINE = '''"""Opt-in checkpoint boundary; fresh-base verification belongs to an external worker."""
import os
from pathlib import Path
from h3_trainer.identity_checkpoint import SOURCE_PINS, export_adapter

PINS = {"sources": dict(SOURCE_PINS), "models": {"transformer_ref": {
    "repo_id": "MiniMaxAI/MiniMax-H3",
    "revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
}}}


def validate_checkpoint_config(config, context):
    if not config.checkpoints.identity_peft_export:
        return
    if (config.model.variant != "ref2va" or config.model.training_mode != "lora"
            or config.model.load_checkpoint is not None
            or config.acceleration.strategy != "ddp"
            or config.acceleration.quantization != "none"
            or context.world_size != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1
            or config.checkpoints.save_training_state != "off"
            or config.lora.rank != 16 or config.lora.alpha != 16
            or config.lora.dropout != 0
            or sorted(config.lora.target_modules) != sorted(("to_q", "to_k", "to_v", "to_out.0"))):
        raise ValueError("identity export requires fresh Ref2VA rank16/alpha16/dropout0, four targets, single-process unquantized ddp and training state off")


def export_checkpoint(model, staging):
    return export_adapter(model, Path(staging) / "identity-peft", pins=PINS)
'''


def plan(source_root):
    """Verify upstream SHA pins, exact contexts and every resulting Python AST."""
    sources = sampling.plan(source_root)
    substitute = sampling.components.guidance.substitute
    config = "src/h3_trainer/config.py"
    sources[config] = substitute(sources[config], "class CheckpointsConfig(ConfigBaseModel):",
        "class CheckpointsConfig(ConfigBaseModel):\n"
        "    identity_peft_export: bool = Field(default=False, strict=True)\n")
    trainer = "src/h3_trainer/trainer.py"
    text = substitute(sources[trainer], "from h3_trainer import logger",
        "from h3_trainer import logger\n"
        "from h3_trainer.identity_checkpoint_pipeline import validate_checkpoint_config")
    text = substitute(text, "    def setup(self) -> None:\n        config = self.config",
        "    def setup(self) -> None:\n        config = self.config\n"
        "        validate_checkpoint_config(config, self.context)")
    text = substitute(text,
        "            save_training_state=config.checkpoints.save_training_state,\n"
        "            config_snapshot=json.loads(config.model_dump_json()),\n"
        "            is_main_process=self.context.is_main,",
        "            save_training_state=config.checkpoints.save_training_state,\n"
        "            config_snapshot=json.loads(config.model_dump_json()),\n"
        "            is_main_process=self.context.is_main,\n"
        "            identity_peft_export=config.checkpoints.identity_peft_export,")
    sources[trainer] = text

    checkpoint = "src/h3_trainer/checkpointing.py"
    text = substitute(sources[checkpoint], "from h3_trainer import logger",
        "from h3_trainer import logger\n"
        "from h3_trainer.identity_checkpoint_pipeline import export_checkpoint")
    text = substitute(text, "    is_main_process: bool = True,\n) -> Path | None:",
        "    is_main_process: bool = True,\n"
        "    identity_peft_export: bool = False,\n) -> Path | None:")
    text = substitute(text, "    weights = gather_trainable_state(model, engine, dtype)",
        "    if identity_peft_export and (engine is not None or save_training_state != 'off'):\n"
        "        raise ValueError('identity export requires unsharded model and training state off')\n"
        "    weights = gather_trainable_state(model, engine, dtype)")
    text = substitute(text,
        "    # os.replace refuses to overwrite a non-empty directory, and the same step can",
        "    if identity_peft_export:\n"
        "        try:\n"
        "            export_checkpoint(model, staging)\n"
        "        except BaseException:\n"
        "            shutil.rmtree(staging, ignore_errors=True)\n"
        "            raise\n\n"
        "    # os.replace refuses to overwrite a non-empty directory, and the same step can")
    sources[checkpoint] = text
    sources["src/h3_trainer/identity_checkpoint.py"] = (
        Path(__file__).parent / "h3_personal/identity_checkpoint.py").read_text()
    sources["src/h3_trainer/identity_checkpoint_pipeline.py"] = PIPELINE
    for name, source in sources.items():
        ast.parse(source, filename=name)
    return sources


def install(source_root, destination):
    """Public worker API: validate first, atomically publish a NEW source-only tree."""
    sources = plan(source_root)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".h3-checkpoint-", dir=destination.parent))
    try:
        for name, source in sources.items():
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(source)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output:
        print(install(args.source, args.output))
    else:
        plan(args.source)
        print("Exact guidance + component + sampling + checkpoint source verified; gate C false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
