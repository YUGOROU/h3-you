"""No devices, network, inference or GPU: streaming protocol + speculative selection."""
import json
import sys
import threading
import unittest
from research.h3_personal.streaming_asr import Endpoint, StreamingASR, SOURCE


def partial(text='こんにちは', revision=1):
    return dict(type='partial', text=text, revision=revision, on_device=True,
                final=False, capture_stopped=False)


def endpoint(text='こんにちは', revision=1):
    return dict(type='utterance_endpoint', text=text, revision=revision,
                on_device=True, final=False, capture_stopped=True,
                endpoint_source=SOURCE, audio_saved=False, cloud_fallback=False)


class StreamingTests(unittest.TestCase):
    def test_endpoint_is_not_apple_final(self):
        state = Endpoint()
        self.assertEqual(state.accept(partial()), 'partial')
        self.assertEqual(state.accept(endpoint()), 'endpoint')
        self.assertIs(state.endpoint['final'], False)
        with self.assertRaises(ValueError):
            state.accept(partial('次', 2))

    def test_protocol_rejects_revisions_sources_and_unstopped_text(self):
        variants = [dict(revision=True), dict(revision=0), dict(final=True),
                    dict(capture_stopped=False), dict(endpoint_source='apple'),
                    dict(text='違う'), dict(audio_saved=True), dict(cloud_fallback=True)]
        for changes in variants:
            with self.subTest(changes=changes):
                state = Endpoint()
                state.accept(partial())
                with self.assertRaises(ValueError):
                    state.accept(dict(endpoint(), **changes))
        state = Endpoint()
        with self.assertRaises(ValueError):
            state.accept(endpoint())
        state.accept(partial())
        with self.assertRaises(ValueError):
            state.accept(partial('別', 1))

    def test_pi_starts_before_endpoint_and_only_latest_is_selected(self):
        started = threading.Event()
        release = threading.Event()
        def propose(text):
            if text == '古い':
                started.set()
                self.assertTrue(release.wait(2))
            return {'reply': text}
        def events():
            yield partial('古い', 1)
            self.assertTrue(started.wait(2))  # actual work before endpoint, not buffering
            yield partial('新しい', 2)
            yield endpoint('新しい', 2)
        with StreamingASR(propose) as stream:
            try:
                asr, proposal = stream.process(events())
                self.assertEqual(asr['revision'], 2)
                self.assertEqual(proposal['reply'], '新しい')
                self.assertFalse(release.is_set())  # stale work cannot block selection
            finally:
                release.set()

    def test_speculation_bounded_and_endpoint_can_select_new_revision(self):
        calls = []
        def propose(text):
            calls.append(text)
            return {'reply': text}
        # Rows are consumed synchronously, so worker threads may not have started yet.
        # The bound is on submitted speculation, not on which threads won the race.
        rows = [partial(str(i), i) for i in range(1, 8)] + [endpoint('7', 7)]
        with StreamingASR(propose) as stream:
            asr, proposal = stream.process(rows)
            submitted = len(stream.futures)
        self.assertEqual(proposal['reply'], '7')
        self.assertEqual(submitted, 2)  # at most two speculative Pi calls
        self.assertIn('7', calls)  # the selected final revision was proposed
        self.assertLessEqual(len(calls), 3)

    def test_missing_endpoint_or_explicit_error_fails_closed(self):
        for rows in ([partial()], [partial(), {'error': 'no final'}]):
            with StreamingASR(lambda text: {'reply': text}) as stream:
                with self.assertRaises(ValueError):
                    stream.process(rows)

    def test_helper_success_required_even_with_endpoint(self):
        rows = [partial(), endpoint()]
        for code in (0, 1):
            script = 'import sys;print(' + repr('\n'.join(json.dumps(r) for r in rows)) + ');sys.exit(' + str(code) + ')'
            with StreamingASR(lambda text: {'reply': text}) as stream:
                if code:
                    with self.assertRaises(ValueError):
                        stream.capture([sys.executable, '-c', script], timeout=2)
                else:
                    asr, proposal = stream.capture([sys.executable, '-c', script], timeout=2)
                    self.assertEqual(proposal['reply'], asr['text'])

    def test_silent_helper_is_terminated_on_deadline(self):
        with StreamingASR(lambda text: {'reply': text}) as stream:
            with self.assertRaises(TimeoutError):
                stream.capture([sys.executable, '-c', 'import time;time.sleep(5)'], timeout=.1)


if __name__ == '__main__':
    unittest.main()
