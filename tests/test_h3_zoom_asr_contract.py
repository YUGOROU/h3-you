import copy
import unittest
from research.h3_personal.asr_source_contract import validate_expected_asr, validate_actual_asr, asr_origin
from research.h3_personal.zoom_input import source_contract
from research.run_h3_relay_conversation import _asr_input
from research.h3_personal.live_relay_worker import validate_turn, PI_MODEL


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(policy='endpoint-native-v1', endpoint_name='h3-asr-t4-7c926aae18', revision='a'*40,
                         endpoint_source='native_completed', completion_policy='input-finalized',
                         capture_stopped=True, cleanup_complete=True)
        self.expected = dict(self.base, input_source=source_contract(reference_sha256='b'*64, capture_source_sha256='c'*64))
        self.actual = dict(self.expected, zoom_capture=dict(schema='h3-zoom-asr-capture-v1',
            profile_sha256=self.expected['input_source']['profile_sha256'], input_epoch=1,
            captured_samples=16000, extracted_samples=16000, submitted_samples=16000, capture_frames=16384,
            capture_stopped=True, capture_cleanup=True, capture_app='zoom', mute_behavior='unmuted', eof=True,
            pending_samples=0, discarded_pending_samples=0, raw_audio_fallback=False, operator_confirmed_remote_idle=True))

    def test_variants(self):
        for actual, expected, origin in ((self.base,self.base,'comet_process_audio_endpoint_asr'),
                (self.actual,self.expected,'zoom_process_voxlock_endpoint_asr')):
            self.assertEqual(validate_actual_asr(actual,expected),actual)
            self.assertEqual(asr_origin(actual),origin)
            self.assertEqual(_asr_input(dict(text='hello',asr=actual)),('hello',actual))
            payload=dict(schema='h3-relay-asr-turn-v1',origin=origin,asr_used=True,asr=actual,
                         director_model=PI_MODEL,reasoning_effort='none',reply='Hello',previous_audio_reused=False)
            self.assertEqual(validate_turn(dict(payload=payload),expected),'Hello')
            with self.assertRaises(ValueError):validate_turn(dict(payload=payload))
            payload['origin']='arbitrary'
            with self.assertRaises(ValueError):validate_turn(dict(payload=payload),expected)

    def test_receipt_mutations(self):
        for key,value in self.actual['zoom_capture'].items():
            for bad in (None, 'false', False if value is not False else True):
                a=copy.deepcopy(self.actual);a['zoom_capture'][key]=bad
                with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):validate_actual_asr(a,self.expected)
        for key,bad in [('submitted_samples',15999),('capture_frames',15999),('captured_samples',720001),('input_epoch',0),('pending_samples',1)]:
            a=copy.deepcopy(self.actual);a['zoom_capture'][key]=bad
            with self.assertRaises(ValueError):validate_actual_asr(a)
        for key in self.actual['zoom_capture']:
            a=copy.deepcopy(self.actual);del a['zoom_capture'][key]
            with self.assertRaises(ValueError):validate_actual_asr(a)

    def test_source_binding(self):
        with self.assertRaises(ValueError):validate_actual_asr(self.actual,self.base)
        with self.assertRaises(ValueError):validate_actual_asr(self.expected)
        a=copy.deepcopy(self.expected);a['input_source']['cpu_threads']=True
        with self.assertRaises(ValueError):validate_expected_asr(a)
        a=copy.deepcopy(self.expected);a['input_source']['reference_sha256']='d'*64
        with self.assertRaises(ValueError):validate_expected_asr(a)
        detached=validate_expected_asr(self.expected);detached['input_source']['model_sha256'].clear()
        self.assertTrue(self.expected['input_source']['model_sha256'])
        for key in ('zoom_capture','extra'):
            a=dict(self.base);a[key]={}
            with self.assertRaises(ValueError):validate_actual_asr(a)


if __name__ == '__main__':unittest.main()
