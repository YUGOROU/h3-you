import hashlib
import tempfile
import unittest
from pathlib import Path
from research.h3_personal.context_parallel_plan import block_plan, verify_source_files, DIFFUSERS_REVISION


class ContextParallelPlanTests(unittest.TestCase):
    def test_only_block_sequence_split_global_outputs_gathered(self):
        p=block_plan(layers=50,sequence_rows=52330,heads=56)
        self.assertEqual(p['transformer_blocks.0']['hidden_states']['split_dim'],1)
        self.assertEqual(p['transformer_blocks.49']['gather_dim'],1)
        self.assertEqual(p['transformer_blocks.*']['adaln_indices']['expected_dims'],1)
        self.assertEqual(len(p['transformer_blocks.*']['rotary_emb']),2)
        self.assertNotIn('token_refiner',p)
        self.assertNotIn('norm_out',p)

    def test_rejects_unsupported_layouts(self):
        for change in [dict(sequence_rows=52331),dict(heads=55),dict(world_size=4),
                       dict(padding_rows=1),dict(layers=1),dict(layers=True)]:
            args=dict(layers=50,sequence_rows=52330,heads=56);args.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):block_plan(**args)

    def test_source_pin_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.py';p.write_bytes(b'audited')
            r=[dict(file='model.py',revision=DIFFUSERS_REVISION,
                    sha256=hashlib.sha256(b'audited').hexdigest())]
            verify_source_files(d,r)
            p.unlink(); p.symlink_to('other.py'); (Path(d)/'other.py').write_bytes(b'audited')
            with self.assertRaises(ValueError):verify_source_files(d,r)
            p.unlink(); p.write_bytes(b'drift')
            with self.assertRaises(ValueError):verify_source_files(d,r)
            r[0]['revision']='main'
            with self.assertRaises(ValueError):verify_source_files(d,r)

    def test_path_escape_and_missing_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verify_source_files(d,[])
            with self.assertRaises(ValueError):verify_source_files(d,[dict(file='../x',revision=DIFFUSERS_REVISION)])
