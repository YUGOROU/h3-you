"""No jobs/weights/network: exact local source + synthetic CPU fixtures only."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research import install_identity_sampling as installer
from research.h3_personal.identity_sampling import IdentityBatchSampler, validate_sampling_config

PIN = Path("/tmp/h3-identity-audit/aviad")
TENSORS = all(importlib.util.find_spec(n) is not None for n in ("torch", "safetensors"))


def dataset():
    return NS(component_index={"version": 2}, split="train", records=[
        NS(id=f"clip-{i:04d}", extras={"quality_pool": "identity" if i < 500 else "motion"},
           bucket_key=(i % 3,), sequence_length=85) for i in range(715)])


def config():
    return NS(data=NS(identity_sampling_budget=715, shuffle=True, val_split_every=0),
              optimization=NS(batch_size=1, max_seq_tokens=85),
              acceleration=NS(strategy="ddp"), seed=42)


class StaticTests(unittest.TestCase):
    def test_coverage_first_exact_once_and_seeded_epochs(self):
        s = IdentityBatchSampler(dataset(), mode='coverage_first', seed=42)
        first = list(s)
        self.assertEqual(len(first),715)
        self.assertEqual(sorted(b[0] for b in first),list(range(715)))
        self.assertEqual(first,list(s))
        s.set_epoch(1)
        second = list(s)
        self.assertNotEqual(first,second)
        self.assertEqual(sorted(b[0] for b in second),list(range(715)))

    def test_coverage_cursor_continues_remaining_epoch_without_replay(self):
        original = IdentityBatchSampler(dataset(), mode='coverage_first', seed=42)
        stream = []
        for epoch in range(3):
            original.set_epoch(epoch)
            stream.extend(list(original))
        for cursor in (0, 25, 714, 715, 716, 1400):
            resumed = IdentityBatchSampler(dataset(), mode='coverage_first', seed=42, start_draw=cursor)
            resumed.set_epoch(0)
            first = list(resumed)
            self.assertEqual(len(resumed), 715 - cursor % 715)
            resumed.set_epoch(1)
            self.assertEqual(len(resumed), 715)
            continuation = first + list(resumed)
            self.assertEqual(continuation, stream[cursor:cursor + len(continuation)])

    def test_sampling_cursor_rejects_bad_types_and_overflow(self):
        for cursor in (-1, True, 715 * 2**63):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                IdentityBatchSampler(dataset(), mode='coverage_first', seed=42, start_draw=cursor)
        c = config()
        c.data.identity_sampling_budget = None
        c.data.identity_sampling_start_draw = 1
        with self.assertRaises(ValueError):
            validate_sampling_config(c, NS(world_size=1, rank=0, local_rank=0))

    @unittest.skipUnless(TENSORS, 'optional torch CPU environment')
    def test_replacement_cursor_replays_seeded_draw_stream_not_model_rng(self):
        original = IdentityBatchSampler(dataset(), seed=42)
        stream = list(original)
        original.set_epoch(1)
        stream += list(original)
        resumed = IdentityBatchSampler(dataset(), seed=42, start_draw=25)
        self.assertEqual(list(resumed), stream[25:715])
        resumed.set_epoch(1)
        self.assertEqual(list(resumed), stream[715:])

    def test_coverage_first_rejects_partial_or_repeated_epoch_budget(self):
        for budget in (714,716,1000):
            with self.assertRaises(ValueError):
                IdentityBatchSampler(dataset(), mode='coverage_first', sample_budget=budget)
        c=config();c.data.identity_sampling_mode='coverage_first';c.data.identity_sampling_budget=1000
        with self.assertRaises(ValueError):
            validate_sampling_config(c,NS(world_size=1,rank=0,local_rank=0))

    def test_metadata_and_contract_fail_closed(self):
        d = dataset()
        s = IdentityBatchSampler(d)
        self.assertEqual(s.pool_counts, {"identity": 500, "motion": 215})
        self.assertEqual(sum(map(len, s.buckets.values())), 715)
        self.assertTrue(all(w > 0 for w in s.weights))
        self.assertAlmostEqual(sum(s.weights[:500]), .65)
        self.assertAlmostEqual(sum(s.weights[500:]), .35)
        self.assertEqual(len(s), 715)
        for kwargs in ({"batch_size": 2}, {"shuffle": False}, {"drop_last": True},
                       {"sample_budget": 0}, {"sample_budget": True}, {"seed": -1},
                       {"max_seq_tokens": 84}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IdentityBatchSampler(d, **kwargs)
        for mutation in (lambda x: x.records.pop(),
                         lambda x: setattr(x, "component_index", None),
                         lambda x: setattr(x, "split", "val"),
                         lambda x: setattr(x.records[0], "id", x.records[1].id),
                         lambda x: x.records[0].extras.clear(),
                         lambda x: x.records[0].extras.update(quality_pool="hold"),
                         lambda x: [r.extras.update(quality_pool="identity") for r in x.records]):
            bad = dataset()
            mutation(bad)
            with self.assertRaises(ValueError):
                IdentityBatchSampler(bad)
        for epoch in (-1, True, 2**63):
            with self.assertRaises(ValueError):
                s.set_epoch(epoch)

    def test_config_guards(self):
        c = config()
        context = NS(world_size=1, rank=0, local_rank=0)
        validate_sampling_config(c, context)
        for target, field, value in ((c.data, "val_split_every", 20),
                                     (c.data, "shuffle", False),
                                     (c.optimization, "batch_size", 2),
                                     (c.acceleration, "strategy", "model_parallel"),
                                     (context, "world_size", 2), (context, "rank", 1),
                                     (context, "local_rank", 1)):
            old = getattr(target, field)
            setattr(target, field, value)
            with self.assertRaises(ValueError):
                validate_sampling_config(c, context)
            setattr(target, field, old)

    @unittest.skipUnless(PIN.exists(), "exact local source unavailable")
    def test_composition_hashes_context_and_atomic_install(self):
        base = installer.components.plan(PIN)
        sources = installer.plan(PIN)
        for name, text in sources.items():
            ast.parse(text, filename=name)
        self.assertEqual(base["src/h3_trainer/datasets.py"], sources["src/h3_trainer/datasets.py"])
        self.assertIn("self.sampler.set_epoch(self.state.epoch)", sources["src/h3_trainer/trainer.py"])
        with tempfile.TemporaryDirectory() as tmp:
            tree = installer.install(PIN, Path(tmp) / "installed")
            with self.assertRaises(FileExistsError):
                installer.install(PIN, tree)
            # A pristine source copy, then a one-byte edit must be rejected before output.
            for name in installer.components.guidance.PINS:
                p = Path(tmp) / "pin" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((PIN / name).read_bytes())
            p.write_bytes(p.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                installer.install(Path(tmp) / "pin", Path(tmp) / "rejected")
            self.assertFalse((Path(tmp) / "rejected").exists())
        bad = dict(base)
        bad["src/h3_trainer/trainer.py"] = bad["src/h3_trainer/trainer.py"].replace(
            "        sampler = BucketBatchSampler(", "        sampler = OtherSampler(")
        with patch.object(installer.components, "plan", return_value=bad):
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                installer.plan(PIN)


@unittest.skipUnless(TENSORS, "torch/safetensors absent; use /tmp/h3-media-env")
class TensorTests(unittest.TestCase):
    def test_replacement_mass_epoch_and_order_reproducibility(self):
        import torch
        d = dataset()
        s = IdentityBatchSampler(d, sample_budget=20000, seed=42)
        torch.manual_seed(999)
        state = torch.get_rng_state().clone()
        first = list(s)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(first, list(s))
        self.assertEqual(len(first), 20000)
        self.assertTrue(all(len(b) == 1 for b in first))
        self.assertEqual(len({b[0] for b in first}), 715)
        identity = sum(d.records[b[0]].extras["quality_pool"] == "identity" for b in first)
        self.assertLess(abs(identity / len(first) - .65), .015)
        s.set_epoch(1)
        self.assertNotEqual(first, list(s))
        s.set_epoch(0)
        self.assertEqual(first, list(s))
        shuffled = copy.deepcopy(d)
        shuffled.records.reverse()
        other = IdentityBatchSampler(shuffled, sample_budget=20000, seed=42)
        self.assertEqual([d.records[b[0]].id for b in first],
                         [shuffled.records[b[0]].id for b in other])

    @unittest.skipUnless(PIN.exists(), "exact local source unavailable")
    def test_actual_patched_setup_data_and_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = installer.install(PIN, Path(tmp) / "trainer")
            result = subprocess.run([sys.executable, "-c", INTEGRATION,
                                     str(tree / "src"), tmp], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


INTEGRATION = r'''
import ast, json, pathlib, sys
from types import SimpleNamespace as NS
import torch
from safetensors.torch import save_file
sys.path.insert(0, sys.argv[1])
from h3_trainer import component_cache as cache
from h3_trainer.datasets import PrecomputedDataset, BucketBatchSampler, bucket_collate
from h3_trainer.config import DataConfig
assert DataConfig(preprocessed_data_root=sys.argv[2]).identity_sampling_budget is None
for value in (0, True, 1.5, "715"):
    try:
        DataConfig(preprocessed_data_root=sys.argv[2], identity_sampling_budget=value)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid configured budget accepted")
assert DataConfig(preprocessed_data_root=sys.argv[2], identity_sampling_budget=715).identity_sampling_budget == 715
from h3_trainer.identity_sampling import build_sampler, validate_sampling_config, IdentityBatchSampler
from torch.utils.data import DataLoader
root = pathlib.Path(sys.argv[2]) / "cache"
root.mkdir()
catalog, refs = {}, {}
def add(kind, spec, tensors, metadata=None):
    path = root / (kind + ".safetensors")
    save_file(tensors, str(path), metadata=metadata)
    entry = dict(kind=kind, path=path.name, sha256=cache.sha256(path), spec=spec,
                 attestation=dict(aviad_revision=cache.REVISION, recipe="SYNTHETIC ONLY",
                                  source_sha256="a"*64, encoder_sha256="b"*64, attested_by="test"))
    key = cache.component_id(entry)
    catalog[key], refs[kind] = entry, key
add("video", dict(width=32,height=32,num_frames=22), {"latents": torch.zeros(7,96)})
add("audio", dict(rows=74), {"latents": torch.zeros(74,32)})
g = [0,1,2,2,0]
add("reference", dict(geometry=g), {"latents": torch.zeros(1,96), "geometry":torch.tensor([g])})
add("text", dict(rows=3), {"prompt_embeds":torch.zeros(3,5120,dtype=torch.bfloat16),
                          "text_token_tags":torch.tensor([1,0,1])})
add("identity_uncond", dict(rows=2,conditional=refs["text"],reference=refs["reference"]),
    {"prompt_embeds":torch.zeros(2,5120,dtype=torch.bfloat16), "text_token_tags":torch.tensor([1,0])},
    dict(schema=cache.PROBE_SCHEMA, aviad_revision=cache.REVISION, probe_prompt=" ",
         conditional_sha256=catalog[refs["text"]]["sha256"],reference_sha256=catalog[refs["reference"]]["sha256"]))
samples = [dict(id=f"clip-{i:04d}", width=32,height=32,num_frames=22,video_rows=7,audio_rows=74,
                text_rows=3,has_audio=True,reference_geometry=[g], components=dict(refs),
                quality_pool="identity" if i<500 else "motion") for i in range(715)]
cache.write_index(root,catalog,samples)
sources = {k:"unused" for k in refs}
# Compile the actual installed method unchanged; omit only heavyweight trainer imports.
# This exercises real setup/data loader wiring without constructing a model or CUDA context.
text = (pathlib.Path(sys.argv[1]) / "h3_trainer/trainer.py").read_text()
trainer = next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="H3Trainer")
method = next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=="_setup_data")
module = ast.Module(body=[method],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),"actual-patched-trainer.py","exec"))
config = NS(data=NS(identity_sampling_budget=1000,shuffle=True,val_split_every=0,
                    num_dataloader_workers=0,preprocessed_data_root=root),
            optimization=NS(batch_size=1,max_seq_tokens=85),seed=42,
            acceleration=NS(strategy="ddp"), training_strategy=NS(data_sources=lambda:sources))
obj = NS(config=config,context=NS(world_size=1,rank=0,local_rank=0),run_logger=NS(message=lambda _:None))
_setup_data(obj)
assert len(obj.train_dataset)==715 and obj.val_dataset is None
assert isinstance(obj.sampler,IdentityBatchSampler) and len(obj.dataloader)==1000
obj.sampler.set_epoch(3)
expected = [obj.train_dataset.records[b[0]].id for b in obj.sampler]
actual = [b["id"] for b in obj.dataloader]
assert expected == actual and len(set(actual)) < len(actual)
b = next(iter(obj.dataloader))
assert b["reference"] is True and b["identity_uncond_text"].shape==(2,5120)
assert b["record"].sequence_length==85 and b["record"].extras["quality_pool"] in ("identity","motion")
try:
    bucket_collate([b,b])
except ValueError:
    pass
else:
    raise AssertionError("batch 2 accepted")
for target, field, value in ((config.data,"val_split_every",20), (config.optimization,"max_seq_tokens",84),
                             (obj.context,"world_size",2)):
    old = getattr(target,field)
    setattr(target,field,value)
    try:
        _setup_data(obj)
    except ValueError:
        pass
    else:
        raise AssertionError(f"guard missing: {field}")
    setattr(target,field,old)
# Budget disabled leaves actual legacy BucketBatchSampler behavior intact.
config.data.identity_sampling_budget=None
_setup_data(obj)
assert isinstance(obj.sampler,BucketBatchSampler) and len(obj.sampler)==715
# Opt-in malformed metadata / cardinality never silently drops or coerces records.
config.data.identity_sampling_budget=1000
p = root / "index.json"
original = json.loads(p.read_text())
for mutate in (lambda x:x["samples"][0].pop("quality_pool"),
               lambda x:x["samples"][0].update(quality_pool="reject"),
               lambda x:(x["samples"].pop(),x.update(num_samples=714))):
    bad=json.loads(json.dumps(original)); mutate(bad); p.write_text(json.dumps(bad))
    try:
        _setup_data(obj)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid eligibility accepted")
p.write_text(json.dumps(original))
try:
    PrecomputedDataset(root,sources,val_split_every=0,max_seq_tokens=84)
except ValueError:
    pass
else:
    raise AssertionError("eligible records silently filtered")
print("actual patched _setup_data + v2 DataLoader: 715 eligible, 1000 draws, guards passed")
'''


if __name__ == "__main__":
    unittest.main()
