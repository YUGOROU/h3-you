import dataclasses
import unittest
import test_h3_streaming_windows as fixtures
from research.h3_personal.streaming_windows import ResamplingContext
from research.h3_personal.streaming_admission import StreamingAdmission, NativeCommitEvidence


def owner(**kw):return StreamingAdmission('c',context=ResamplingContext('same-rate-32k'),**kw)


def commit(controller,ticket):
    audio=ticket.decision.audio
    controller.commit(ticket,NativeCommitEvidence(ticket.decision.window.end,
        None if audio is None else audio.prefix_sha256,'b'*64))


def drain(controller,ticket):
    commit(controller,ticket)
    controller.ack_decoded(ticket,ticket.decision.window.end-(0 if ticket.decision.conversation_final else 5))


def idle(controller):
    decision=controller.inspect();assert decision.role=='listening'
    ticket=controller.reserve(decision);drain(controller,ticket);return ticket


class AdmissionTests(unittest.TestCase):
    def test_audio_arriving_in_reserved_idle_drains_then_speech(self):
        c=owner();stream=fixtures.source();c.attach(stream)
        ticket=c.reserve(c.inspect())
        fixtures.append_samples(stream,136000)
        self.assertEqual(c.inspect().status,'busy')
        self.assertEqual(ticket.decision.role,'listening')
        drain(c,ticket)
        decision=c.inspect();self.assertEqual(decision.role,'speech')
        self.assertEqual(decision.audio.onset_frame,124)
        speech=c.reserve(decision);drain(c,speech)
        self.assertEqual(c.snapshot()['committed_frame'],226)
        self.assertEqual(c.snapshot()['speech_onset_frame'],124)
        self.assertFalse(c.snapshot()['native_runtime_connected'])

    def test_idle_keeps_running_while_partial_audio_waits(self):
        c=owner();stream=fixtures.source();c.attach(stream)
        idle(c);fixtures.append_samples(stream,100)
        self.assertEqual(c.inspect().role,'listening')
        idle(c)
        fixtures.append_samples(stream,135900)
        d=c.inspect();self.assertEqual(d.role,'speech')
        self.assertEqual(d.audio.onset_frame,226)
        self.assertEqual(d.window.start,204)

    def test_stale_idle_inspection_refused_if_speech_becomes_ready(self):
        c=owner();idle(c);stream=fixtures.source();c.attach(stream)
        decision=c.inspect();self.assertEqual(decision.role,'listening')
        fixtures.append_samples(stream,136000)
        with self.assertRaises(ValueError):c.reserve(decision)
        self.assertFalse(c.snapshot()['reserved'])

    def test_starvation_inside_speech_never_becomes_idle(self):
        c=owner();idle(c);stream=fixtures.source();fixtures.append_samples(stream,136000);c.attach(stream)
        drain(c,c.reserve(c.inspect()))
        d=c.inspect();self.assertEqual((d.status,d.role),('wait','speech'))
        self.assertEqual(c.snapshot()['committed_frame'],226)
        fixtures.append_samples(stream,136000)
        self.assertEqual(c.inspect().status,'ready')

    def test_eos_retirement_return_to_idle_and_final_only_flush(self):
        c=owner();idle(c);stream=fixtures.source();fixtures.append_samples(stream,32000);fixtures.seal(stream);c.attach(stream)
        t=c.reserve(c.inspect());commit(c,t)
        with self.assertRaises(RuntimeError):c.finish_speech()
        c.ack_decoded(t,221)
        self.assertEqual(c.inspect().status,'end');c.finish_speech()
        self.assertEqual(c.inspect().role,'listening')
        c.request_stop();t=c.reserve(c.inspect());commit(c,t)
        c.ack_decoded(t,328)
        self.assertEqual(c.snapshot()['phase'],'closed')
        with self.assertRaises(RuntimeError):c.inspect()

    def test_wrong_proof_duplicate_and_foreign_ticket(self):
        c=owner();t=c.reserve(c.inspect())
        with self.assertRaises(ValueError):c.commit(dataclasses.replace(t),NativeCommitEvidence(124,None,'b'*64))
        with self.assertRaises(ValueError):c.ack_decoded(t,119)
        commit(c,t)
        with self.assertRaises(ValueError):commit(c,t)
        self.assertEqual(c.inspect().status,'busy')
        c.ack_decoded(t,119)
        with self.assertRaises(ValueError):c.ack_decoded(t,119)

    def test_cancelled_source_not_published_after_reservation(self):
        c=owner();idle(c);stream=fixtures.source();fixtures.append_samples(stream,136000);c.attach(stream)
        t=c.reserve(c.inspect());stream.cancel()
        with self.assertRaises(RuntimeError):commit(c,t)
        self.assertEqual(c.snapshot()['committed_frame'],124)
        self.assertEqual(c.snapshot()['phase'],'failed')

    def test_uncertain_native_commit_and_decode_are_terminal(self):
        c=owner();t=c.reserve(c.inspect())
        with self.assertRaises(ValueError):c.commit(t,NativeCommitEvidence(125,None,'b'*64))
        self.assertEqual(c.snapshot()['phase'],'failed')
        with self.assertRaises(RuntimeError):commit(c,t)
        for final in (False,True):
            c=owner()
            if final:c.request_stop()
            t=c.reserve(c.inspect());commit(c,t)
            with self.assertRaises(ValueError):c.ack_decoded(t,119 if final else 124)
            self.assertEqual(c.snapshot()['phase'],'failed')
            with self.assertRaises(RuntimeError):c.ack_decoded(t,124 if final else 119)

    def test_cancel_pending_only_not_native_speech(self):
        c=owner();idle(c);stream=fixtures.source();c.attach(stream);c.cancel_pending_speech()
        with self.assertRaises(RuntimeError):stream.snapshot()
        c.request_stop();self.assertTrue(c.inspect().conversation_final)
        c=owner();idle(c);stream=fixtures.source();fixtures.append_samples(stream,136000);c.attach(stream)
        c.reserve(c.inspect())
        with self.assertRaises(RuntimeError):c.cancel_pending_speech()

    def test_source_epochs_voice_format_and_attempts_are_bounded(self):
        c=owner(max_turns=2);idle(c)
        a=fixtures.source(epoch=5);c.attach(a);c.cancel_pending_speech()
        for b in (fixtures.source(epoch=4,name='late'),fixtures.source(epoch=6,name='speech')):
            with self.assertRaises(ValueError):c.attach(b)
        b=fixtures.source(epoch=6,name='new');c.attach(b);c.cancel_pending_speech()
        with self.assertRaises(ValueError):c.attach(fixtures.source(epoch=7,name='third'))

    def test_idle_limit_final_slot_and_deadline_do_not_reset(self):
        c=owner(max_idle_windows=1,max_windows=3);idle(c)
        self.assertEqual(c.inspect().status,'limit')
        c.request_stop();drain(c,c.reserve(c.inspect()))
        self.assertEqual(c.snapshot()['phase'],'closed')
        now=[0];c=owner(clock=lambda:now[0],timeout=1);idle(c);now[0]=1
        with self.assertRaises(TimeoutError):c.inspect()
        self.assertEqual(c.snapshot()['phase'],'failed')

    def test_second_turn_same_warm_controller_new_source_only(self):
        c=owner(max_turns=2);idle(c)
        for epoch in (1,2):
            s=fixtures.source(epoch=epoch,name=str(epoch));fixtures.append_samples(s,100);fixtures.seal(s);c.attach(s)
            drain(c,c.reserve(c.inspect()));c.finish_speech()
            idle(c)
        self.assertEqual(c.snapshot()['completed_turns'],2)
        self.assertEqual(c.snapshot()['committed_frame'],532)
