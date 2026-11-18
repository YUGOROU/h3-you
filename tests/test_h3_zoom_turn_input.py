"""Device/cloud-free attended pipeline regressions."""
import io
import threading
import time
import unittest
from unittest.mock import patch
from research.h3_personal.zoom_input import CaptureInterrupted, InputBlocked, source_contract
from research.h3_personal.zoom_turn_input import BoundedRaw, ZoomTurnInput, validate_zoom_capture


class ZoomTurnTests(unittest.TestCase):
    def source(self):
        return source_contract(reference_sha256='a'*64, capture_source_sha256='b'*64)

    def test_bounded_raw_does_not_consume_backlog(self):
        raw = io.BytesIO(b'ab'*20)
        bounded = BoundedRaw(raw, 8)
        self.assertEqual(bounded.read(100), b'ab'*8)
        self.assertEqual(bounded.read(100), b'')
        self.assertEqual(raw.tell(), 16)

    def test_early_eof_is_capture_interruption_not_decline(self):
        with self.assertRaises(CaptureInterrupted):
            BoundedRaw(io.BytesIO(), 8).read(10)
        self.assertTrue(issubclass(CaptureInterrupted, InputBlocked))

    def test_reader_premature_stream_end_interrupts_without_replay(self):
        p = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                          all_participants_consented=True, tail_validator=lambda **kw: True)
        p.prepare_capture(deadline=time.monotonic()+20)
        import struct
        raw = BoundedRaw(io.BytesIO(struct.pack('<100h', *([1000]*100))), 8*16000)
        from research.h3_personal.zoom_input import ZoomPCMReader
        class Extractor:
            def reset(inner): pass
            def push(inner, x): return x * 0.5
            def flush(inner): return []
        reader = ZoomPCMReader(raw, Extractor(), p.guard, p.epoch)
        self.assertEqual(reader.read(640), struct.pack('<100h', *([500]*100)))
        with self.assertRaises(CaptureInterrupted):
            reader.read(640)
        self.assertTrue(reader.failed)
        self.assertEqual(reader.stats['discarded_pending_samples'], 0)
        self.assertEqual(reader.receipt()['failed'], True)

    def test_gate_requires_consent_and_fresh_confirmation(self):
        p = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                          tail_validator=lambda **kw: True)
        with self.assertRaises(InputBlocked):
            p.prepare_capture(deadline=time.monotonic()+20)
        p.consented = True
        p.prepare_capture(deadline=time.monotonic()+20)
        epoch = p.epoch
        p.guard.check(epoch)
        p.note_reply_published()
        with self.assertRaises(InputBlocked):
            p.guard.check(epoch)
        p.confirmation = lambda **kw: False
        with self.assertRaises(InputBlocked):
            p.prepare_capture(deadline=time.monotonic()+20)

    def test_live_arm_requires_validated_echo_tail(self):
        # Default contract: no operator-judgement bypass of the measured tail.
        p = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                          all_participants_consented=True)
        with self.assertRaisesRegex(InputBlocked, 'validated remote echo tail'):
            p.prepare_capture(deadline=time.monotonic()+20)
        seen = {}
        def validator(*, deadline, cancel):
            seen.update(deadline=deadline, cancel=cancel)
            return False
        p2 = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                           all_participants_consented=True, tail_validator=validator)
        with self.assertRaisesRegex(InputBlocked, 'not validated'):
            p2.prepare_capture(deadline=time.monotonic()+20)
        self.assertIn('cancel', seen)
        # Validator runs before the operator confirmation; a refused tail never
        # reaches the TTY prompt.
        prompts = []
        def confirm(**kw):
            prompts.append(kw)
            return True
        p3 = ZoomTurnInput(object(), self.source(), confirmation=confirm,
                           all_participants_consented=True,
                           tail_validator=lambda **kw: False)
        with self.assertRaises(InputBlocked):
            p3.prepare_capture(deadline=time.monotonic()+20)
        self.assertEqual(prompts, [])
        p4 = ZoomTurnInput(object(), self.source(), confirmation=confirm,
                           all_participants_consented=True,
                           tail_validator=lambda **kw: True)
        p4.prepare_capture(deadline=time.monotonic()+20)
        self.assertEqual(len(prompts), 1)
        with self.assertRaises(ValueError):
            ZoomTurnInput(object(), self.source(), tail_validator='x')
        with self.assertRaises(ValueError):
            ZoomTurnInput(object(), self.source(), tail_validation_required='no')

    def test_cancel_and_send_during_confirmation(self):
        cancel = threading.Event()
        p = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                          all_participants_consented=True, cancel=cancel,
                          tail_validator=lambda **kw: True)
        cancel.set()
        with self.assertRaises(InputBlocked):
            p.prepare_capture(deadline=time.monotonic()+20)
        cancel.clear()
        def confirm(**kwargs):
            p.note_reply_published()
            return True
        p.confirmation = confirm
        with self.assertRaises(InputBlocked):
            p.prepare_capture(deadline=time.monotonic()+20)

    def test_tail_validation_invalidation_never_prompts_or_rearms(self):
        for action in ('reply', 'cancel', 'expire'):
            with self.subTest(action=action):
                prompts = []
                p = ZoomTurnInput(object(), self.source(),
                    confirmation=lambda **kw: prompts.append(kw) or True,
                    all_participants_consented=True)
                clock = [100.0]
                def validate(**kw):
                    if action == 'reply':
                        p.note_reply_published()
                    elif action == 'cancel':
                        p.cancel.set()
                    else:
                        clock[0] = 121.0
                    return True
                p.tail_validator = validate
                with patch('research.h3_personal.zoom_turn_input.time.monotonic', side_effect=lambda: clock[0]):
                    with self.assertRaises(InputBlocked):
                        p.prepare_capture(deadline=120.0)
                self.assertEqual(prompts, [])
                self.assertTrue(p.guard.closed)

    def test_expired_or_cancelled_turn_does_not_call_validator(self):
        from unittest.mock import Mock
        for cancelled in (False, True):
            validator = Mock(return_value=True)
            p = ZoomTurnInput(object(), self.source(), all_participants_consented=True,
                              tail_validator=validator)
            if cancelled:
                p.cancel.set()
            with self.assertRaises(InputBlocked):
                p.prepare_capture(deadline=time.monotonic() + (20 if cancelled else -1))
            validator.assert_not_called()
            self.assertTrue(p.guard.closed)

    def test_exact_max_asr_boundary_forces_eof_flush(self):
        p = ZoomTurnInput(object(), self.source(), confirmation=lambda **kw: True,
                          all_participants_consented=True, tail_validator=lambda **kw: True)
        p.prepare_capture(deadline=time.monotonic()+20)
        class Reader:
            def read(self, size):
                self.drained = True
                return b''
            def receipt(self):
                return dict(input_epoch=2, captured_samples=128000, extracted_samples=128000,
                            submitted_samples=128000, eof=self.drained, pending_samples=0,
                            discarded_pending_samples=0, raw_audio_fallback=False, failed=False)
        p.reader = Reader()
        evidence = p.finish_capture(dict(type='stopped', frames=128010, cleanup=True,
                                        app='zoom', mute_behavior='unmuted'),
                                    dict(audio_sent_seconds=8), capture_seconds=8)
        self.assertTrue(evidence['eof'])
        for key, bad in [('capture_frames', 127999), ('capture_frames', 720001),
                         ('eof', False), ('pending_samples', 1), ('capture_app', 'comet'),
                         ('raw_audio_fallback', True), ('submitted_samples', 127999)]:
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_zoom_capture(dict(evidence, **{key:bad}), self.source())
        with self.assertRaises(ValueError):
            validate_zoom_capture(dict(evidence, extra=True), self.source())


if __name__ == '__main__':
    unittest.main()
