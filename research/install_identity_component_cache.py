#!/usr/bin/env python3
"""Compose a real shared-path loader onto the exact pinned guidance installer."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import tempfile

try:
    from research import install_identity_guidance as guidance
except ModuleNotFoundError:
    import install_identity_guidance as guidance


def plan(source_root):
    sources = guidance.plan(source_root)
    path = "src/h3_trainer/datasets.py"
    text = sources[path]
    text = guidance.substitute(text,
        '        return self.text_rows + self.audio_rows + self.video_rows',
        '        extras = self.extras or {}\n'
        '        references = sum(g[1] * (g[2] // 2) * (g[3] // 2) + 2 * g[4]\n'
        '                         for g in extras.get("reference_geometry", []))\n'
        '        text_rows = max(self.text_rows, extras.get("identity_probe_text_rows", self.text_rows))\n'
        '        return text_rows + self.audio_rows + self.video_rows + references')
    text = guidance.substitute(text,
        '        records = [SampleRecord.from_dict(entry) for entry in payload["samples"]]',
        '        self.component_index = None\n'
        '        if payload.get("version") == 2:\n'
        '            from h3_trainer.component_cache import validate_index\n'
        '            self.component_index = validate_index(self.root, payload)\n'
        '            if set(self.data_sources) != {"video", "audio", "text", "reference", "identity_uncond"}:\n'
        '                raise ValueError("v2 requires the exact guidance data-source subset")\n'
        '            for sample in payload["samples"]:\n'
        '                sample["identity_probe_text_rows"] = payload["components"][sample["components"]["identity_uncond"]]["spec"]["rows"]\n'
        '            self.component_samples = {s["id"]: s for s in payload["samples"]}\n'
        '        records = [SampleRecord.from_dict(entry) for entry in payload["samples"]]')
    text = guidance.substitute(text, '        directory = self.data_sources.get(source)',
        '        if self.component_index is not None:\n'
        '            from h3_trainer.component_cache import load_component\n'
        '            return load_component(self, source, sample_id, required)\n'
        '        directory = self.data_sources.get(source)')
    text = guidance.substitute(text, '        # Vision blocks embedded in the prompt',
        '        if self.component_index is not None:\n'
        '            item["component_cache_v2"] = True\n'
        '        # Vision blocks embedded in the prompt')
    text = guidance.substitute(text, '    if any("identity_uncond_text" in item for item in batch):',
        '    if any("identity_uncond_text" in item or item.get("component_cache_v2") for item in batch):')
    text = guidance.substitute(text, '            if len(kept) != len(records):',
        '            if len(kept) != len(records) and self.component_index is not None:\n'
        '                raise ValueError("v2 adopted samples exceed reference/probe-inclusive max_seq_tokens")\n'
        '            if len(kept) != len(records):')
    sources[path] = text
    trainer = 'src/h3_trainer/trainer.py'
    sources[trainer] = guidance.substitute(sources[trainer], '        self._skipped_long = 0',
        '        self._skipped_long = 0\n        self._skipped_total = 0')
    runtime = (Path(__file__).parent / "h3_personal/identity_guidance.py").read_text()
    runtime = guidance.substitute(runtime, '    root = dataset.root\n    path = root / dataset.data_sources["identity_uncond"]',
        '    if getattr(dataset, "component_index", None) is not None:\n'
        '        from h3_trainer.component_cache import load_probe\n'
        '        return load_probe(dataset, record)\n'
        '    root = dataset.root\n    path = root / dataset.data_sources["identity_uncond"]')
    # Keep the legacy fixed-ID loader/metadata untouched for v1 trees.
    sources["src/h3_trainer/identity_guidance.py"] = runtime
    sources["src/h3_trainer/component_cache.py"] = (Path(__file__).parent / "h3_personal/component_cache.py").read_text()
    for name, source in sources.items():
        ast.parse(source, filename=name)
    return sources


def install(source_root, destination):
    sources = plan(source_root)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".h3-components-", dir=destination.parent))
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
        print("Exact guidance + component source verified; gate C false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
