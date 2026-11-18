import hashlib
import json
import math
import random
import struct
import unittest
import zlib

from research.h3_personal import pcm_transport_experiment as exp


def fixtures():
    rng = random.Random(731)
    return {
        "silence": bytes(96000),
        "speech_like": b"".join(struct.pack("<h", int(
            10000 * math.sin(i * 0.031) * (0.5 + 0.5 * math.sin(i * 0.0007))
        )) for i in range(48000)),
        "incompressible": rng.randbytes(96000),
    }


class PCMTransportTests(unittest.TestCase):
    def mutate(self, record, index, value, payload=None):
        header = list(exp._HEADER.unpack_from(record))
        header[index] = value
        return exp._HEADER.pack(*header) + (record[exp._HEADER.size:] if payload is None else payload)

    def test_benchmark_real_timings_and_integrity(self):
        for name, pcm in fixtures().items():
            with self.subTest(name=name):
                report = exp.benchmark_pcm(pcm, repeats=2)
                json.dumps(report, allow_nan=False)
                self.assertEqual(report["sha256"], hashlib.sha256(pcm).hexdigest())
                self.assertEqual(report["sample_count"], 48000)
                for row in report["results"]:
                    self.assertGreater(row["encode_seconds_mean"], 0)
                    self.assertGreater(row["decode_seconds_mean"], 0)
                    self.assertEqual(row["decoded_sha256"], report["sha256"])
                    self.assertEqual(row["decoded_sample_count"], 48000)
                for level in (None, 1, 3, 6):
                    self.assertEqual(exp._decode_record(exp._encode_record(pcm, level=level)), pcm)

    def test_input_bounds(self):
        for pcm in (b"", b"a", bytes(exp.MAX_PCM_BYTES + 2), bytearray(2)):
            with self.assertRaises(ValueError):
                exp.benchmark_pcm(pcm)
        for repeats in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                exp.benchmark_pcm(bytes(2), repeats=repeats)
        for rate, channels in ((0, 1), (48000, 0), (True, 1), (48000, 33), (48000, 2)):
            with self.assertRaises(ValueError):
                exp.benchmark_pcm(bytes(2), rate, channels)
        with self.assertRaises(ValueError):
            exp._encode_record(bytes(2), level=9)

    def test_max_and_stereo(self):
        pcm = bytes(exp.MAX_PCM_BYTES)
        self.assertEqual(exp._decode_record(exp._encode_record(pcm, channels=2)), pcm)
        report = exp.benchmark_pcm(bytes(16), channels=2, repeats=3)
        self.assertEqual(report["sample_count"], 8)
        self.assertEqual(report["frame_count"], 4)

    def test_strict_records(self):
        record = exp._encode_record(bytes(100))
        payload = record[exp._HEADER.size:]
        invalid = [b"", record[:-1], record + b"x",
                   bytes(exp._HEADER.size + exp.MAX_COMPRESSED_BYTES + 1)]
        for index, value in ((0, b"BAD!"), (1, 2), (2, 3), (3, 9), (4, 0),
                             (5, 0), (6, 49), (7, exp.MAX_PCM_BYTES + 1),
                             (8, exp.MAX_COMPRESSED_BYTES + 1), (9, bytes(32))):
            invalid.append(self.mutate(record, index, value))
        for bad_payload in (payload + b"trailing", payload + payload, payload[:-1], b"garbage",
                            zlib.compress(bytes(exp.MAX_PCM_BYTES + 100))):
            invalid.append(self.mutate(record, 8, len(bad_payload), bad_payload))
        raw = exp._encode_record(bytes(100), level=None)
        invalid.append(raw[:-1] + b"x")
        for bad in invalid:
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                exp._decode_record(bad)


if __name__ == "__main__":
    unittest.main()
