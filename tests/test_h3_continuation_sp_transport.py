"""Pinned native SP transport regression; no Torch/GPU or model execution."""
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from research.h3_personal import fast_continuation as bridge
from research.h3_personal.continuation_plan import plan
from test_h3_fast_continuation_stages import pinned_source


class Vector:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = 'float32'
        self.device = 'cpu'

    def dim(self):
        return 1

    def __getitem__(self, key):
        return Vector(self.values[key[0] if isinstance(key, tuple) else key])


def native_functions(path, names, namespace):
    source = ast.parse(path.read_text())
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError('pinned native functions missing')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


class TransportTests(unittest.TestCase):
    def test_native_odd_even_shard_gather_preserves_all_rows(self):
        with pinned_source() as root:
            packing = types.SimpleNamespace(__file__=str(root/'fastvideo/pipelines/basic/minimax_h3/packing.py'))
            for size in (1, 2, 415, 416, 55551, 55552):
                bridge.verify_sequence_transport(packing, size, 2)
            torch = types.SimpleNamespace(full=lambda shape, value, **kw: Vector([value]*shape[0]),
                                          cat=lambda parts, dim: Vector(sum((p.values for p in parts), [])))
            group = types.SimpleNamespace(rank=0)
            group.shard = lambda value, dim, scale_grad: Vector(value.values[group.rank*len(value.values)//2:(group.rank+1)*len(value.values)//2])
            ns = dict(torch=torch, get_sp_world_size=lambda: 2, get_sp_group=lambda: group)
            native_functions(root/'fastvideo/distributed/utils.py',
                             ['compute_padding_for_sp', 'pad_sequence_tensor', 'unpad_sequence_tensor'], ns)
            native_functions(root/'fastvideo/distributed/communication_op.py',
                             ['sequence_model_parallel_shard', 'sequence_model_parallel_all_gather_with_unpad'], ns)
            for size in (1, 2, 415, 416, 55551, 55552):
                original = Vector(range(size))
                shards = []
                for rank in (0, 1):
                    group.rank = rank
                    shard, original_length = ns['sequence_model_parallel_shard'](original, dim=0)
                    self.assertEqual(original_length, size)
                    shards.append(shard)
                self.assertEqual(shards[0].shape, shards[1].shape)
                group.all_gather = lambda value, dim: Vector(shards[0].values+shards[1].values)
                restored = ns['sequence_model_parallel_all_gather_with_unpad'](shards[0], size, dim=0)
                self.assertEqual(restored.values, original.values)
                self.assertEqual(original.values, list(range(size)))
            # Hash-bound attention/model evidence: no dummy logical tokens reach attention/output.
            attention = (root/'fastvideo/attention/layer.py').read_text()
            model = (root/'fastvideo/models/dits/minimax_h3.py').read_text()
            self.assertIn('qkv = qkv[:, :original_seq_len, :, :]', attention)
            self.assertIn('original_seq_len = sequence_length', model)
            self.assertIn('sequence_model_parallel_all_gather_with_unpad(video_output, original_seq_len', model)
            self.assertIn('sequence_model_parallel_all_gather_with_unpad(audio_output, original_seq_len', model)

    def test_transport_source_tamper_and_invalid_lengths_rejected(self):
        with pinned_source() as root:
            packing = types.SimpleNamespace(__file__=str(root/'fastvideo/pipelines/basic/minimax_h3/packing.py'))
            for size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    bridge.verify_sequence_transport(packing, size, 2)
            path = root/'fastvideo/distributed/communication_op.py'
            path.write_text(path.read_text()+'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'SP transport differs'):
                bridge.verify_sequence_transport(packing, 415, 2)

    def test_layout_odd_length_is_not_changed_or_rejected(self):
        # Execute real build_layout with tensor-operation doubles. This catches
        # the deployed modulus guard, not numerical rotary/attention parity.
        @dataclass
        class Layout:
            sequence_length: int
            position_ids: object
            video_indices: object
            audio_indices: object
        tags = MagicMock()
        tags.__eq__.return_value = MagicMock()
        tags.ndim = 1
        tags.dtype = 'long'
        tags.numel.return_value = 3
        torch = types.SimpleNamespace(long='long', isfinite=lambda p: types.SimpleNamespace(all=lambda: True))
        image = types.SimpleNamespace(media_type='image', num_latent_frames=1, has_audio=False,
                                      num_audio_latents=0, latent_height=2, latent_width=2)
        window = plan(328, 124, 22)[0]
        for length in (415, 416):
            layout = Layout(length, MagicMock(), MagicMock(), MagicMock())
            packing = types.SimpleNamespace(build_ref2va_packed_sequence=MagicMock(return_value=layout))
            with patch.dict(sys.modules, {'torch': torch}), patch.object(bridge, 'verify_packing'), patch.object(bridge, 'verify_sequence_transport') as verified:
                result = bridge.build_layout(packing, object(), text_tags=tags, images=[image], window=window, height=2, width=2, ranks=2)
            verified.assert_called_once_with(packing, length, 2)
            self.assertEqual(result.sequence_length, length)
            self.assertIs(result.video_indices, layout.video_indices)
            self.assertIs(result.audio_indices, layout.audio_indices)
            self.assertIs(packing.build_ref2va_packed_sequence.call_args.args[0], tags.cpu())


if __name__ == '__main__':
    unittest.main()
