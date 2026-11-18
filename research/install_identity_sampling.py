#!/usr/bin/env python3
"""Compose opt-in weighted sampling atop the exact component-cache plan."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import tempfile

try:
    from research import install_identity_component_cache as components
except ModuleNotFoundError:
    import install_identity_component_cache as components


def plan(source_root):
    sources = components.plan(source_root)
    substitute = components.guidance.substitute
    config = "src/h3_trainer/config.py"
    sources[config] = substitute(sources[config], "class DataConfig(ConfigBaseModel):",
        "class DataConfig(ConfigBaseModel):\n"
        "    identity_sampling_start_draw: int = Field(default=0, ge=0, strict=True)\n"
        "    identity_sampling_budget: int | None = Field(default=None, ge=1, strict=True)\n"
        "    identity_sampling_mode: Literal['replacement', 'coverage_first'] = 'replacement'\n")
    trainer = "src/h3_trainer/trainer.py"
    text = substitute(sources[trainer], "from h3_trainer.datasets import BucketBatchSampler, PrecomputedDataset, bucket_collate",
        "from h3_trainer.datasets import BucketBatchSampler, PrecomputedDataset, bucket_collate\n"
        "from h3_trainer.identity_sampling import build_sampler, validate_sampling_config")
    text = substitute(text, "    def _setup_data(self) -> None:\n        config = self.config",
        "    def _setup_data(self) -> None:\n        config = self.config\n"
        "        validate_sampling_config(config, self.context)")
    text = substitute(text,
        "        sampler = BucketBatchSampler(\n"
        "            self.train_dataset,\n"
        "            batch_size=config.optimization.batch_size,\n"
        "            shuffle=config.data.shuffle,\n"
        "            seed=config.seed,\n"
        "        )",
        "        sampler = build_sampler(self.train_dataset, config, BucketBatchSampler)")
    sources[trainer] = text
    sources["src/h3_trainer/identity_sampling.py"] = (
        Path(__file__).parent / "h3_personal/identity_sampling.py").read_text()
    for name, source in sources.items():
        ast.parse(source, filename=name)
    return sources


def install(source_root, destination):
    sources = plan(source_root)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".h3-sampling-", dir=destination.parent))
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
        print("Exact component + sampling source verified; gate C false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
