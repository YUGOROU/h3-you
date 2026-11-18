"""CPU-only accepted runtime admission and policy identity regressions."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.h3_personal.accepted_profiles import choose_profile, AcceptedQualityProfile
from research.h3_personal import conditioning_experiment as ce
from research.h3_personal.live_voice_runtime import LiveVoiceRuntime


class AcceptedRuntimeTests(unittest.TestCase):
    def test_strict_record(self):
        p = choose_profile(ordered_source_sha256=('a'*64,), prompt_sha256='b'*64)
        self.assertEqual(AcceptedQualityProfile.from_record(p.wire_record()), p)
        for change in ({'short_edge': 512}, {'unexpected': 1}, {'fps': True},
                       {'expected_rt': [1, 2]}):
            with self.assertRaises(ValueError):
                AcceptedQualityProfile.from_record(dict(p.wire_record(), **change))
        record = p.wire_record()
        del record['prompt_sha256']
        with self.assertRaises(ValueError):
            AcceptedQualityProfile.from_record(record)

    def test_runtime_binding_switch_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory).resolve()
            ref = root/'ref'
            ref.write_bytes(b'reference')
            p = choose_profile(ordered_source_sha256=(hashlib.sha256(ref.read_bytes()).hexdigest(),),
                               prompt_sha256=hashlib.sha256(b'prompt').hexdigest())
            kwargs = {k: ref for k in ('model', 'model_manifest', 'identity', 'identity_config',
                                      'smd', 'reference', 'archive', 'output')}
            self.assertIsNone(LiveVoiceRuntime(**kwargs).quality_profile)
            with self.assertRaises(ValueError):
                LiveVoiceRuntime(**kwargs, quality_profile=p)
            owner = LiveVoiceRuntime(**kwargs, continuation=True, quality_profile=p)
            owner.config = SimpleNamespace(prompt='prompt', width=256, height=448, num_frames=124, fps=24)
            owner._state = 'ready'
            owner.continuation_file = root/'continuation.json'
            owner.continuation_file.write_text('{"mode":"independent"}')
            owner.conditioning_file = root/'policy.json'
            owner.quality_profile_file = root/'accepted-quality-profile.json'
            ce._publish_private(owner.conditioning_file, p.legacy_conditioning_manifest())
            ce._publish_private(owner.quality_profile_file, p.wire_record())
            os.environ[ce.EXPERIMENT_ENV] = str(owner.conditioning_file)
            os.environ[ce.ACCEPTED_PROFILE_ENV] = str(owner.quality_profile_file)
            key = ce.policy_key()
            token = ce._edge.set(512)
            try:
                precomputed_key = ce.policy_key()
            finally:
                ce._edge.reset(token)
            with self.assertRaises(RuntimeError):
                owner.configure_conditioning(512)
            q = dataclasses.replace(p, name='reference512')
            owner.configure_quality_profile(q)
            self.assertEqual(precomputed_key, ce.policy_key())
            self.assertNotEqual(key, ce.policy_key())
            with self.assertRaises(RuntimeError):
                owner.configure_quality_profile(dataclasses.replace(q, prompt_sha256='c'*64))
            owner._validate_quality_binding()
            owner.config.prompt = 'drift'
            with self.assertRaises(ValueError):
                owner._validate_quality_binding()
            owner.config.prompt = 'prompt'
            ref.write_bytes(b'drift')
            with self.assertRaises(ValueError):
                owner._validate_quality_binding()
            owner.close()
            self.assertNotIn(ce.ACCEPTED_PROFILE_ENV, os.environ)
            self.assertNotIn(ce.EXPERIMENT_ENV, os.environ)

    def test_actual_window_binding_and_native_clear(self):
        from test_h3_conditioning_experiment import actual_token
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            root = Path(d).resolve()
            source = root/'image'
            source.write_bytes(b'image')
            p = choose_profile(ordered_source_sha256=(hashlib.sha256(b'image').hexdigest(),),
                               prompt_sha256=hashlib.sha256(b'prompt').hexdigest())
            policy, accepted = root/'policy', root/'accepted'
            ce._publish_private(policy, p.legacy_conditioning_manifest())
            ce._publish_private(accepted, p.wire_record())
            os.environ.update({ce.EXPERIMENT_ENV: str(policy), ce.ACCEPTED_PROFILE_ENV: str(accepted)})
            token = actual_token(1)
            batch = SimpleNamespace(prompt='prompt', width=256, height=448, num_frames=124, fps=24,
                references=[SimpleNamespace(source=source, media_type='image')], extra={})
            ce.snapshot_sources(batch)
            batch.extra['_h3_continuation'] = {'token': token}
            batch.references = [SimpleNamespace(media_type='image', image=SimpleNamespace(size=(1024,1824)))]
            ce.capture_prepared_geometry(batch)
            batch.references = None  # native latent-preparation lifecycle
            self.assertEqual(ce._receipt_sizes(batch), [[1024,1824]])
            ce.validate_accepted_batch(batch)
            for attribute, value in [('prompt', 'different'), ('width', 512)]:
                old = getattr(batch, attribute)
                setattr(batch, attribute, value)
                with self.assertRaises(ValueError):
                    ce.validate_accepted_batch(batch)
                setattr(batch, attribute, old)
            stage = SimpleNamespace(processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda: {})))
            with self.assertRaises(ValueError):
                ce.validate_accepted_batch(batch, stage)
            source.write_bytes(b'drift')
            with self.assertRaises(RuntimeError):
                ce.validate_accepted_batch(batch)

    def test_legacy_without_accepted_file(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            path = Path(d)/'policy.json'
            path.write_text(json.dumps(dict(schema=ce.SCHEMA, short_edges=[1024,512], active_short_edge=1024)))
            os.environ[ce.EXPERIMENT_ENV] = str(path)
            self.assertIsNone(ce.read_accepted_profile())
            self.assertIsNotNone(ce.policy_key())


if __name__ == '__main__':
    unittest.main()
