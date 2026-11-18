import copy
import unittest

from research.h3_personal.identity_training_adapter import SCALE_RULE, audit_ref2va_v1_header, subject_parameter_scope


def fixture():
    header = {"__metadata__": {"ss_base_model_version": "minimax_h3_ref2va"}}
    offset = 0
    for family, count in (("blocks", 50), ("token_refiner.blocks", 2)):
        for i in range(count):
            for name, in_dim, out_dim in (("attn.qkv_proj", 5376, 21504),
                                          ("attn.out_proj", 7168, 5376),
                                          ("mlp.fc1", 5376, 28672),
                                          ("mlp.fc2", 14336, 5376)):
                for part, shape in (("A", [16, in_dim]), ("B", [out_dim, 16])):
                    size = 2 * shape[0] * shape[1]
                    header[f"diffusion_model.{family}.{i}.{name}.lora_{part}.weight"] = dict(
                        dtype="BF16", shape=shape, data_offsets=[offset, offset+size])
                    offset += size
    return header


class TrainingAdapterAuditTests(unittest.TestCase):
    def audit(self, header):
        return audit_ref2va_v1_header(header, scale_rule=SCALE_RULE)

    def test_full_mapping_no_mutation(self):
        h = fixture()
        original = copy.deepcopy(h)
        p = self.audit(h)
        self.assertEqual(h, original)
        self.assertEqual(len(p), 312)
        self.assertEqual(len({x.target for x in p}), 312)
        self.assertEqual({x.scale for x in p}, {1.0})
        self.assertEqual({x.rank for x in p}, {16})
        self.assertTrue(any(x.target == 'token_refiner.refiner_blocks.1.ff.net.2' for x in p))

    def test_qkv_shared_a_exact_output_slices(self):
        p = self.audit(fixture())[:3]
        self.assertEqual(len({x.source_a for x in p}), 1)
        self.assertEqual(len({x.source_b for x in p}), 1)
        self.assertEqual([x.b_rows for x in p], [(0, 7168), (7168, 14336), (14336, 21504)])
        self.assertEqual([x.target for x in p], [f'transformer_blocks.0.attn.to_{s}' for s in 'qkv'])
        self.assertEqual({x.input_features for x in p}, {5376})
        self.assertEqual({x.output_features for x in p}, {7168})

    def test_unknown_scaling_rejected(self):
        for rule in (None, 'rank16-implies-alpha16', ''):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                audit_ref2va_v1_header(fixture(), scale_rule=rule)

    def test_missing_extra_and_alpha_keys_rejected(self):
        for change in ('missing', 'extra', 'alpha'):
            h = fixture()
            if change == 'missing':
                del h[next(k for k in h if k != '__metadata__')]
            else:
                h['unexpected.alpha' if change == 'alpha' else 'unexpected.weight'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit(h)

    def test_bad_shape_dtype_and_offsets_rejected(self):
        for changes in ({'shape': [16, 5376]}, {'dtype': 'F32'},
                        {'data_offsets': [0, 1]}, {'data_offsets': [True, 229376]},
                        {'data_offsets': [2, 229378]}):
            h = fixture()
            key = 'diffusion_model.blocks.0.attn.out_proj.lora_A.weight'
            h[key].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.audit(h)

    def test_wrong_base_marker_rejected(self):
        h = fixture()
        h['__metadata__']['ss_base_model_version'] = 'minimax_h3_fl2va'
        with self.assertRaises(ValueError):
            self.audit(h)


class SubjectScopeTests(unittest.TestCase):
    def names(self):
        stems = [f'transformer_blocks.{i}' for i in range(50)]
        stems += [f'token_refiner.refiner_blocks.{i}' for i in range(2)]
        return [f'{stem}.attn.{p}.lora_{ab}.default.weight'
                for stem in stems for p in ('to_q', 'to_k', 'to_v', 'to_out.0') for ab in 'AB']

    def test_window_keeps_full_canonical_keys_but_freezes_refiner(self):
        names = self.names()
        scope = subject_parameter_scope(names)
        self.assertEqual(len(scope.trainable), 240)
        self.assertEqual(len(scope.frozen), 176)
        self.assertEqual(set(scope.trainable) | set(scope.frozen), set(names))
        self.assertFalse(set(scope.trainable) & set(scope.frozen))
        self.assertTrue(all('token_refiner' not in x for x in scope.trainable))
        self.assertTrue(any('transformer_blocks.20.' in x for x in scope.trainable))
        self.assertTrue(any('transformer_blocks.19.' in x for x in scope.frozen))

    def test_supported_prefix_and_order_independence(self):
        names = ['base_model.model.' + x for x in self.names()]
        self.assertEqual(subject_parameter_scope(names), subject_parameter_scope(reversed(names)))

    def test_wrong_adapter_missing_duplicate_and_invalid_windows_rejected(self):
        names = self.names()
        for invalid in (names[:-1], names + [names[0]],
                        [x.replace('.default.', '.training_adapter.') for x in names]):
            with self.assertRaises(ValueError):
                subject_parameter_scope(invalid)
        for start, end in ((True, 50), (20, 20), (-1, 50), (20, 51)):
            with self.assertRaises(ValueError):
                subject_parameter_scope(names, start_block=start, end_block=end)


if __name__ == '__main__':
    unittest.main()
