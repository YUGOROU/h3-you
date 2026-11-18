import copy
import unittest
from research.h3_personal.benchmark_analysis import analyze_batch


def fixture():
    results = []
    for i in range(3):
        results.append(dict(case=i, latent_file_sha256='a'*64,
            generation=dict(schema='h3-inference-profile-v1', phase='generate', precision='bf16',
                quantization='none', denoiser_forwards=49, video_seconds=5,
                generation_seconds=90 if i == 0 else 80, denoise_seconds=70,
                phases={'load_seconds':10}, condition_cache_hit=i>0, model_resident_hit=i==2),
            decode=dict(schema='h3-inference-profile-v1', phase='decode', decode_seconds=10,
                vae_load_seconds=2, vae_decode_seconds=7)))
    return dict(schema='h3-same-condition-resident-batch-v1', results=results,
                exact_latent_file_bytes_equal=True)


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_nested_totals_not_double_counted(self):
        r = analyze_batch(fixture())
        self.assertEqual(r['results'][0]['render_seconds'],100)
        self.assertEqual(r['results'][0]['generation_other_seconds'],10)
        self.assertAlmostEqual(r['cold_to_last_render_speedup'],100/90)
        self.assertFalse(r['subjective_identity_verified'])
        self.assertFalse(r['causal_streaming_verified'])

    def test_bad_numeric_and_baseline_rejected(self):
        for key, value in [('generation_seconds',float('nan')),('denoise_seconds',-1),
                           ('video_seconds',True),('precision','fp8'),('denoiser_forwards',8)]:
            with self.subTest(key=key):
                r=fixture();r['results'][0]['generation'][key]=value
                with self.assertRaises(ValueError):analyze_batch(r)

    def test_parity_contradiction_and_phase_overlap_rejected(self):
        r=fixture();r['results'][2]['latent_file_sha256']='b'*64
        with self.assertRaises(ValueError):analyze_batch(r)
        r['exact_latent_file_bytes_equal']=False
        self.assertFalse(analyze_batch(r)['exact_latent_file_bytes_equal'])
        r=fixture();r['results'][0]['decode']['vae_decode_seconds']=100
        with self.assertRaises(ValueError):analyze_batch(r)

    def test_order_geometry_and_hash_rejected(self):
        for target,key,value in [('outer','case',1),('outer','latent_file_sha256','bad'),
                                 ('generation','video_seconds',6)]:
            r=copy.deepcopy(fixture());obj=r['results'][0]
            if target!='outer':obj=obj[target]
            obj[key]=value
            with self.assertRaises(ValueError):analyze_batch(r)
