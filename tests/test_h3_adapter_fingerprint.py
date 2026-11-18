"""Stdlib-only fake safetensors fixtures; no torch, network or devices."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from research.h3_personal.adapter_fingerprint import fingerprint_file, SCHEMA


def tensor(dtype='U8', shape=None, offsets=None):
    return {'dtype': dtype, 'shape': [2] if shape is None else shape,
            'data_offsets': [0, 2] if offsets is None else offsets}


class FingerprintTests(unittest.TestCase):
    def fingerprint(self, header, payload=b'abcd'):
        raw = header if isinstance(header, bytes) else json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fake.safetensors'
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
            return fingerprint_file(path)

    def test_order_metadata_and_offsets_are_not_identity(self):
        first = {'a': tensor(), 'b': tensor(offsets=[2, 4]),
                 '__metadata__': {'note': 'first'}}
        second = {'__metadata__': {'different': 'ignored'},
                  'b': tensor(), 'a': tensor(offsets=[2, 4])}
        a = self.fingerprint(first)
        self.assertEqual(a, self.fingerprint(second, b'cdab'))
        self.assertEqual(a['canonical_tensor_schema'], SCHEMA)
        self.assertNotEqual(a, self.fingerprint(first, b'abce'))
        for replacement in ({'x': tensor(), 'b': tensor(offsets=[2, 4])},
                            {'a': tensor('I8'), 'b': tensor(offsets=[2, 4])},
                            {'a': tensor(shape=[1, 2]), 'b': tensor(offsets=[2, 4])}):
            self.assertNotEqual(a, self.fingerprint(replacement))

    def test_invalid_descriptors(self):
        for bad in (tensor(offsets=[-1, 1]), tensor(offsets=[0, 5]),
                    tensor(offsets=[2, 0]), tensor(offsets=[False, 2]),
                    tensor(offsets=[0]), tensor(shape=[-1]),
                    tensor(shape=[True]), tensor(shape=[1.0]),
                    tensor(shape='2'), tensor('UNKNOWN'), tensor('F32'),
                    [], {'dtype': 'U8', 'shape': [2]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.fingerprint({'a': bad}, b'ab')

    def test_overlap_gaps_trailing_and_metadata(self):
        for header in ({'a': tensor(), 'b': tensor(offsets=[1, 3])},
                       {'a': tensor(offsets=[1, 3])}, {'a': tensor()},
                       {'__metadata__': {'x': 4}}, {'__metadata__': None}):
            with self.subTest(header=header), self.assertRaises(ValueError):
                self.fingerprint(header)

    def test_bad_json_and_duplicate_keys(self):
        for raw in (b'{"a":{},"a":{}}',
                    b'{"a":{"dtype":"U8","dtype":"I8"}}',
                    b'{"__metadata__":{"x":"a","x":"b"}}',
                    b'{"x":NaN}', b'[]', b'{', b'{"x":"\xff"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.fingerprint(raw)

    def test_truncated_and_oversized_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad'
            for raw in (b'', b'123', struct.pack('<Q', 100) + b'{}',
                        struct.pack('<Q', 2**64 - 1)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    fingerprint_file(path)

    def test_scalars_empty_and_chunked_payload(self):
        self.fingerprint({'scalar': tensor('I16', [], [0, 2]),
                          'empty': tensor(shape=[0], offsets=[0, 0])}, b'ab')
        payload = b'x' * (1024 * 1024 + 3)
        self.fingerprint({'large': tensor(shape=[len(payload)], offsets=[0, len(payload)])}, payload)
        self.fingerprint({}, b'')


if __name__ == '__main__':
    unittest.main()
