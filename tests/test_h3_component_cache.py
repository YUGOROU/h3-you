"""Synthetic CPU tensors only. Run tensor tests in /tmp/h3-media-env."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from research.h3_personal import component_cache as cache
from research import install_identity_component_cache as installer

PIN = Path("/tmp/h3-identity-audit/aviad")
TENSORS = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))


class StaticTests(unittest.TestCase):
    def test_identity_and_provenance(self):
        entry = {"kind": "video", "path": "one", "sha256": "a" * 64}
        other = dict(entry, path="two")
        self.assertEqual(cache.component_id(entry), cache.component_id(other))
        self.assertEqual(cache.provenance_status(entry), "unresolved")
        self.assertFalse(cache.C_GATE_ENABLED)
        with self.assertRaises(ValueError):
            cache.geometry_rows(32, 32, 23)

    def test_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.write_text("x")
            (root / "escape").symlink_to(outside)
            for path in (str(outside), "../outside", "escape"):
                with self.assertRaises(ValueError):
                    cache.contained_path(root, path)

    @unittest.skipUnless(PIN.exists(), "local exact source fixture unavailable")
    def test_patch_and_pins(self):
        sources = installer.plan(PIN)
        for name, source in sources.items():
            ast.parse(source, filename=name)
        self.assertIn("return load_probe(dataset, record)", sources["src/h3_trainer/identity_guidance.py"])
        self.assertIn('f"{record.id}.safetensors"', sources["src/h3_trainer/identity_guidance.py"])
        with tempfile.TemporaryDirectory() as tmp:
            output = installer.install(PIN, Path(tmp) / "installed")
            self.assertTrue((output / "src/h3_trainer/component_cache.py").exists())
            with self.assertRaises(FileExistsError):
                installer.install(PIN, output)
            with self.assertRaises(ValueError):
                installer.plan(tmp)


@unittest.skipUnless(TENSORS, "torch/safetensors absent; use separate /tmp/h3-media-env")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = {}
        self.refs = {}
        self.torch, self.save = torch, save_file
        vr, ar = cache.geometry_rows(32, 32, 22)
        g = [0, 1, 2, 2, 0]
        self.add("video", {"width": 32, "height": 32, "num_frames": 22}, {"latents": torch.zeros(vr, 96)})
        self.add("audio", {"rows": ar}, {"latents": torch.zeros(ar, 32)})
        self.add("reference", {"geometry": g}, {"latents": torch.zeros(1, 96), "geometry": torch.tensor([g])})
        self.add("text", {"rows": 3}, {"prompt_embeds": torch.zeros(3, 5120, dtype=torch.bfloat16),
                                           "text_token_tags": torch.tensor([1, 0, 1])})
        metadata = {"schema": cache.PROBE_SCHEMA, "aviad_revision": cache.REVISION, "probe_prompt": " ",
                    "conditional_sha256": self.catalog[self.refs["text"]]["sha256"],
                    "reference_sha256": self.catalog[self.refs["reference"]]["sha256"]}
        self.add("identity_uncond", {"rows": 2, "conditional": self.refs["text"], "reference": self.refs["reference"]},
                 {"prompt_embeds": torch.zeros(2, 5120, dtype=torch.bfloat16), "text_token_tags": torch.tensor([1, 0])}, metadata)
        self.samples = [{"id": sid, "width": 32, "height": 32, "num_frames": 22, "video_rows": vr,
                         "audio_rows": ar, "text_rows": 3, "has_audio": True, "caption": "synthetic",
                         "reference_geometry": [g], "components": dict(self.refs)} for sid in ("a", "b")]

    def add(self, kind, spec, tensors, metadata=None):
        # Arbitrary shared paths, deliberately not <source>/<sample-id>.
        path = self.root / (kind + ".safetensors")
        self.save(tensors, str(path), metadata=metadata)
        entry = {"kind": kind, "path": path.name, "sha256": cache.sha256(path), "spec": spec,
                 "attestation": {"aviad_revision": cache.REVISION, "recipe": "SYNTHETIC TEST ONLY",
                                 "source_sha256": "a" * 64, "encoder_sha256": "b" * 64, "attested_by": "unit test"}}
        key = cache.component_id(entry)
        self.catalog[key] = entry
        self.refs[kind] = key

    def test_write_fresh_and_complete(self):
        output = cache.write_index(self.root, self.catalog, self.samples)
        payload = json.loads(output.read_text())
        self.assertFalse(payload["c_gate_enabled"])
        self.assertEqual(len(payload["components"]), 5)
        with self.assertRaises(FileExistsError):
            cache.write_index(self.root, self.catalog, self.samples)
        for field, value in (("has_audio", False), ("text_rows", 0), ("audio_rows", 1), ("reference_geometry", [])):
            bad = copy.deepcopy(payload)
            bad["samples"][0][field] = value
            with self.assertRaises(ValueError):
                cache.validate_index(self.root, bad)
        bad = copy.deepcopy(payload)
        del bad["samples"][0]["components"]["identity_uncond"]
        with self.assertRaises(ValueError):
            cache.validate_index(self.root, bad)
        bad = copy.deepcopy(payload)
        bad["samples"][1]["id"] = "a"
        with self.assertRaises(ValueError):
            cache.validate_index(self.root, bad)
        bad = copy.deepcopy(payload)
        bad["samples"][0]["id"] = "../unsafe"
        with self.assertRaises(ValueError):
            cache.validate_index(self.root, bad)
        summaries=cache.validate_catalog(self.root,self.catalog)
        self.assertTrue(all(set(s)=={"vision_rows"} for s in summaries.values()))
        bad = copy.deepcopy(payload)
        probe_key = self.refs["identity_uncond"]
        probe = bad["components"].pop(probe_key)
        probe["spec"]["conditional"] = self.refs["audio"]
        new_key = cache.component_id(probe)
        bad["components"][new_key] = probe
        for sample in bad["samples"]:
            sample["components"]["identity_uncond"] = new_key
        with self.assertRaises(ValueError):
            cache.validate_index(self.root, bad)

    def test_tensor_rejections(self):
        torch = self.torch
        original = self.catalog[self.refs["video"]]
        for tensor in (torch.zeros(7, 95), torch.zeros(7, 96, dtype=torch.float16),
                       torch.full((7, 96), float("nan"))):
            self.save({"latents": tensor}, str(self.root / original["path"]))
            entry = dict(original, sha256=cache.sha256(self.root / original["path"]))
            with self.assertRaises(ValueError):
                cache.validate_catalog(self.root, {cache.component_id(entry): entry})

    def test_hash_and_unresolved(self):
        entry = self.catalog[self.refs["video"]]
        unresolved = dict(entry, attestation=None)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            cache.validate_catalog(self.root, {cache.component_id(unresolved): unresolved})
        with (self.root / entry["path"]).open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash"):
            cache.validate_catalog(self.root, self.catalog)

    def test_tags_geometry_and_metadata(self):
        for kind, tensors in (
            ("text", {"prompt_embeds": self.torch.zeros(3, 5120, dtype=self.torch.bfloat16),
                      "text_token_tags": self.torch.tensor([1, 2, 0])}),
            ("reference", {"latents": self.torch.zeros(1, 96), "geometry": self.torch.tensor([[0, 2, 2, 2, 0]])}),
        ):
            entry = self.catalog[self.refs[kind]]
            self.save(tensors, str(self.root / entry["path"]))
            updated = dict(entry, sha256=cache.sha256(self.root / entry["path"]))
            with self.assertRaises(ValueError):
                cache.validate_catalog(self.root, {cache.component_id(updated): updated})
        # Rebuild the damaged fixtures before checking probe metadata.
        self.setUp()
        probe = self.catalog[self.refs["identity_uncond"]]
        from safetensors.torch import load_file
        tensors = load_file(str(self.root / probe["path"]))
        self.save(tensors, str(self.root / probe["path"]), metadata={"sample_id": "a"})
        updated = dict(probe, sha256=cache.sha256(self.root / probe["path"]))
        del self.catalog[self.refs["identity_uncond"]]
        self.catalog[cache.component_id(updated)] = updated
        with self.assertRaisesRegex(ValueError, "metadata"):
            cache.validate_catalog(self.root, self.catalog)

    @unittest.skipUnless(PIN.exists(), "local exact source fixture unavailable")
    def test_real_patched_dataset_shared_paths(self):
        cache.write_index(self.root, self.catalog, self.samples)
        tree = installer.install(PIN, self.root / "trainer")
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from h3_trainer.datasets import PrecomputedDataset, bucket_collate
sources = {k: "unused-fixed-id-directory" for k in ("video", "audio", "text", "reference", "identity_uncond")}
d = PrecomputedDataset(sys.argv[2], sources, val_split_every=0)
a, b = d[0], d[1]
assert a["video"].shape == (7, 96)
assert a["audio"].shape == (74, 32)
assert a["identity_uncond_text"].shape == (2, 5120)
assert a["reference"] is True
assert a["reference_video_rows"].shape == (1, 96)
assert a["text"].equal(b["text"])
assert a["record"].sequence_length == 85  # 3 text + 74 audio + 7 target + 1 reference
try:
    PrecomputedDataset(sys.argv[2], sources, val_split_every=0, max_seq_tokens=84)
except ValueError:
    pass
else:
    raise AssertionError("v2 silently dropped adopted samples")
# The composed installer must retain the actual v1 guidance fixed-ID contract.
import json, pathlib, shutil
from safetensors.torch import load_file, save_file
from h3_trainer.identity_guidance import write_uncond_cache
root = pathlib.Path(sys.argv[2])
legacy = root / "legacy"
legacy.mkdir()
legacy_sources = {"video": "latents", "audio": "audio_latents", "text": "conditions", "reference": "reference_latents", "identity_uncond": "identity_uncond"}
for kind, directory in legacy_sources.items():
    if kind == "identity_uncond":
        continue
    (legacy / directory).mkdir()
    shutil.copyfile(root / (kind + ".safetensors"), legacy / directory / "a.safetensors")
probe = load_file(str(root / "identity_uncond.safetensors"))
write_uncond_cache(legacy, "a", legacy / "conditions/a.safetensors", probe["prompt_embeds"], probe["text_token_tags"])
sample = json.loads((root / "index.json").read_text())["samples"][0]
(legacy / "index.json").write_text(json.dumps({"version": 1, "samples": [sample]}))
old = PrecomputedDataset(legacy, legacy_sources, val_split_every=0)[0]
assert old["identity_uncond_text"].equal(a["identity_uncond_text"])
try:
    bucket_collate([a, b])
except ValueError:
    pass
else:
    raise AssertionError("batch 2 accepted")
# Loading must catch changed bytes after initialization, too.
with open(sys.argv[2] + "/audio.safetensors", "ab") as f:
    f.write(b"changed")
try:
    d[0]
except ValueError:
    pass
else:
    raise AssertionError("changed bytes accepted")
'''
        result = subprocess.run([sys.executable, "-c", script, str(tree / "src"), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
