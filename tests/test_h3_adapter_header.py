import json
import struct
import unittest
from research.h3_personal.adapter_header import inspect_header, header_length, MAX_HEADER_BYTES


def fixture():
    return {'__metadata__':{'alpha':'8'},
            'block.attn.lora_A.default.weight':{'shape':[128,5376]},
            'block.attn.lora_B.default.weight':{'shape':[5376,128]}}


def inspect(value):
    body=json.dumps(value).encode();return inspect_header(struct.pack('<Q',len(body)),body)


class AdapterHeaderTests(unittest.TestCase):
    def test_rank_does_not_imply_alpha(self):
        r=inspect(fixture());self.assertEqual(r['peft_modules'][0]['rank'],128)
        self.assertEqual(r['metadata']['alpha'],'8');self.assertFalse(r['alpha_verified'])
        self.assertFalse(r['identity_composition_verified'])

    def test_incomplete_and_rank_mismatch(self):
        r=fixture();del r['block.attn.lora_B.default.weight']
        with self.assertRaises(ValueError):inspect(r)
        r=fixture();r['block.attn.lora_B.default.weight']['shape'][1]=64
        with self.assertRaises(ValueError):inspect(r)

    def test_limits_duplicate_and_truncation(self):
        with self.assertRaises(ValueError):header_length(struct.pack('<Q',MAX_HEADER_BYTES+1))
        with self.assertRaises(ValueError):inspect_header(b'123',b'{}')
        b=b'{"x":{},"x":{}}'
        with self.assertRaises(ValueError):inspect_header(struct.pack('<Q',len(b)),b)
        with self.assertRaises(ValueError):inspect_header(struct.pack('<Q',3),b'{}')

    def test_custom_head_not_mislabelled_peft(self):
        r=fixture();r['pdd.heads']={'shape':[32,96,5376]}
        self.assertEqual(inspect(r)['other_tensor_names'],['pdd.heads'])
