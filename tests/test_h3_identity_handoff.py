import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from research.h3_personal import identity_checkpoint as cp, identity_handoff as ho
from research.h3_personal.identity_handoff_fetch import fetch_handoff
from research import hf_identity_train as worker
try:
    import torch
except ImportError:
    torch = None


class RecipeTests(unittest.TestCase):
    def test_digest_stable_but_changes_with_recipe(self):
        self.assertEqual(ho.recipe_digest({'a': 1, 'b': 2}), ho.recipe_digest({'b': 2, 'a': 1}))
        self.assertNotEqual(ho.recipe_digest({'a': 1}), ho.recipe_digest({'a': 2}))
        with self.assertRaises(ValueError):
            ho.recipe_digest({'a': float('nan')})


@unittest.skipIf(torch is None, 'optional torch CPU environment')
class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'checkpoint-0000025' / 'identity-peft'
        self.pins = {'sources': cp.SOURCE_PINS, 'models': {'transformer_ref': {'repo_id': 'test/synthetic', 'revision': 'a'*40}}}
        def save(state, path):
            Path(path).write_text(json.dumps({k: v.tolist() for k, v in state.items()}))
        def load(path, **kw):
            return {k: torch.tensor(v) for k, v in json.loads(Path(path).read_text()).items()}
        def setter(model, state, **kw):
            model.state = {k: v.clone() for k, v in state.items()}
            return NS(missing_keys=['base.weight'], unexpected_keys=[])
        self.boundary = cp.Boundary(lambda m, **kw: m.state, setter, save, load, torch)
        self.model = self.fresh()
        self.model.state['layer.lora_B.weight'].fill_(1)
        cp.export_adapter(self.model, self.path, pins=self.pins, boundary=self.boundary)
        self.recipe = ho.recipe_digest({'objective': 'synthetic'})
        self.metadata = ho.escrow_metadata(self.path, pins=self.pins, recipe_sha256=self.recipe, run_id='modal-run', step=25)

    def fresh(self):
        config = dict(peft_type='LORA', r=16, lora_alpha=16, lora_dropout=0,
                      target_modules=list(cp.TARGETS), bias='none')
        return NS(peft_config={'default': NS(to_dict=lambda: copy.deepcopy(config))}, active_adapters=['default'],
                  state={'layer.lora_A.weight': torch.zeros(16, 3), 'layer.lora_B.weight': torch.zeros(2, 16)})

    def test_exported_escrow_never_claims_verified_or_exact_resume(self):
        self.assertFalse((self.path / cp.VERIFIED).exists())
        self.assertFalse(self.metadata['quality_verified'])
        self.assertFalse(self.metadata['exact_resume'])
        self.assertFalse(self.metadata['optimizer_state_present'])

    def test_fresh_strict_weight_reload_and_lineage(self):
        fresh = self.fresh()
        lineage = ho.restore_weights(fresh, self.path, self.metadata, pins=self.pins,
                                     expected_recipe_sha256=self.recipe, boundary=self.boundary)
        self.assertEqual(lineage['parent_step'], 25)
        self.assertTrue(lineage['new_optimizer'])
        self.assertFalse(lineage['exact_resume'])
        self.assertTrue(torch.equal(fresh.state['layer.lora_B.weight'], self.model.state['layer.lora_B.weight']))
        cp.inspect_verified(self.path, expected_pins=self.pins)

    def test_bound_cursor_config_continuation_and_tamper_rejection(self):
        r = self.request()
        cursor = dict(consumed_draws=745, mode='replacement', epoch_budget=715, seed=42, index_sha256='b'*64)
        metadata = ho.escrow_metadata(self.path, pins=self.pins, recipe_sha256=worker.training_recipe_digest(r),
                                     run_id='modal-run', step=25, sampling_cursor=cursor)
        path = Path(self.tmp.name) / 'HANDOFF.json'
        path.write_text(json.dumps(metadata))
        warm = dict(r, warm_start=dict(checkpoint=str(self.path), metadata=str(path), metadata_sha256=cp.sha256(path)))
        self.assertEqual(worker.handoff_sampling_start(warm), 745)
        self.assertEqual(worker.config_payload(warm)['data']['identity_sampling_start_draw'], 745)
        self.assertEqual(worker.training_recipe_digest(r), worker.training_recipe_digest(warm))
        with self.assertRaises(ValueError):
            worker.handoff_sampling_start(dict(warm, sampling_mode='coverage_first'))
        path.write_text('{}')
        with self.assertRaises(ValueError):
            worker.handoff_sampling_start(warm)

    def test_cursor_invalid_draws_seed_and_index_rejected(self):
        cursor = dict(consumed_draws=25, mode='replacement', epoch_budget=715, seed=42, index_sha256='b'*64)
        for changes in (dict(consumed_draws=True), dict(consumed_draws=24), dict(epoch_budget=714),
                        dict(seed=-1), dict(index_sha256='bad')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ho.validate_sampling_cursor(dict(cursor, **changes), minimum_draws=25)

    def test_bad_recipe_artifact_flags_and_parent_rejected(self):
        for mutate in ('recipe', 'artifact', 'optimizer', 'parent'):
            metadata = copy.deepcopy(self.metadata)
            if mutate == 'recipe': metadata['recipe_sha256'] = '0'*64
            elif mutate == 'artifact': metadata['artifacts'][cp.WEIGHTS] = '0'*64
            elif mutate == 'optimizer': metadata['optimizer_state_present'] = True
            else: metadata['parent_step'] = True
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                ho.validate_handoff(self.path, metadata, pins=self.pins, expected_recipe_sha256=self.recipe)

    def request(self):
        return dict(model='/model', cache='/cache', max_updates=100, work=self.tmp.name, run_id='modal-run',
                    pins=self.pins, index_sha256='b'*64, output_repo='test/private')

    def test_target_budget_and_paths_not_recipe_but_optimizer_and_sampling_are(self):
        r = self.request()
        digest = worker.training_recipe_digest(r)
        other = dict(r, work='/different', max_updates=200)
        self.assertEqual(digest, worker.training_recipe_digest(other))
        self.assertNotEqual(digest, worker.training_recipe_digest(dict(r, learning_rate=0.01)))
        self.assertNotEqual(digest, worker.training_recipe_digest(dict(r, sampling_mode='coverage_first')))

    def test_private_atomic_escrow_namespace_offline_restored(self):
        calls = []
        constants = NS(HF_HUB_OFFLINE=True)
        fake_api = NS(repo_info=lambda repo: NS(private=True),
                      create_commit=lambda **kw: (calls.append(kw) or NS(oid='c'*40)))
        fake_module = NS(HfApi=lambda: fake_api, CommitOperationAdd=lambda **kw: NS(**kw), constants=constants)
        with patch.dict('sys.modules', {'huggingface_hub': fake_module}):
            worker.upload_escrow(self.request(), self.path, 25)
            worker.upload_escrow(self.request(), self.path, 25)  # same bound bytes: no duplicate commit
        self.assertTrue(constants.HF_HUB_OFFLINE)
        self.assertEqual(len(calls), 1)
        operations = calls[0]['operations']
        self.assertEqual(len(operations), 5)
        self.assertTrue(all('/escrow/checkpoint-0000025/' in op.path_in_repo for op in operations))
        receipt = json.loads((Path(self.tmp.name) / 'latest-escrow.json').read_text())
        self.assertEqual(receipt['revision'], 'c'*40)
        self.assertFalse(receipt['exact_resume'])

    def fetch(self, *, info=None, digest=None, revision='d'*40):
        path = Path(self.tmp.name) / 'source-HANDOFF.json'
        path.write_text(json.dumps(self.metadata))
        calls = []
        def download(**kw):
            calls.append(kw)
            name = Path(kw['filename']).name
            return path if name == 'HANDOFF.json' else self.path / name
        result = fetch_handoff(repo_id='test/private', revision=revision,
            prefix='identity/modal-run/escrow/checkpoint-0000025', metadata_sha256=digest or cp.sha256(path),
            destination=Path(self.tmp.name) / 'fetched', pins=self.pins, expected_recipe_sha256=self.recipe,
            repo_info=lambda **kw: info or NS(private=True, sha='d'*40), download=download)
        return result, calls

    def test_private_revision_fetch_atomic_and_source_unmodified(self):
        result, calls = self.fetch()
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(c['revision'] == 'd'*40 for c in calls))
        self.assertFalse((Path(result['checkpoint']) / cp.VERIFIED).exists())
        cp.inspect_export(result['checkpoint'], expected_pins=self.pins)
        self.assertTrue(self.path.is_dir())
        self.assertFalse((self.path / cp.VERIFIED).exists())
        with self.assertRaises(FileExistsError):
            self.fetch()

    def test_fetch_public_revision_and_metadata_rejected_without_partial_tree(self):
        for kwargs in (dict(info=NS(private=False, sha='d'*40)), dict(info=NS(private=True, sha='e'*40)),
                       dict(digest='0'*64), dict(revision='main')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fetch(**kwargs)
            self.assertFalse((Path(self.tmp.name) / 'fetched').exists())
            self.assertEqual(list(Path(self.tmp.name).glob('.h3-handoff-fetch-*')), [])

    def test_public_repository_rejected_without_commit(self):
        constants = NS(HF_HUB_OFFLINE=True)
        fake_api = NS(repo_info=lambda repo: NS(private=False))
        fake_module = NS(HfApi=lambda: fake_api, CommitOperationAdd=lambda **kw: NS(**kw), constants=constants)
        with patch.dict('sys.modules', {'huggingface_hub': fake_module}), self.assertRaises(ValueError):
            worker.upload_escrow(self.request(), self.path, 25)
        self.assertTrue(constants.HF_HUB_OFFLINE)


if __name__ == '__main__':
    unittest.main()
