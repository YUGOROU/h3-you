import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from research.run_h3_browser_conversation import main, own


class OwnerTests(unittest.TestCase):
    def exercise(self, mode='ok'):
        events = []
        now = [0.0]
        holder = {}

        class Server:
            def __init__(self, root, status, **kwargs):
                self.root = root
                self.status = status
                self.client_ready = threading.Event()
                self.playback_done = threading.Event()
                self.stream_state = 'waiting'
                self.url = 'http://127.0.0.1/private'
                self.kwargs = kwargs
                holder['server'] = self
                events.append('created')

            def __enter__(self):
                events.append('enter')
                return self

            def __exit__(self, *exc):
                events.append('closed')

            def validate_output(self):
                events.append('validate')
                if not self.status()['default_is_builtin_speaker']:
                    raise RuntimeError('output fail')

        def sleep(seconds):
            now[0] += seconds
            server = holder['server']
            if mode == 'cancel':
                server.stream_state = 'failed'
            elif mode != 'timeout':
                events.append('ready')
                server.client_ready.set()

        def sink(root, binding, **kwargs):
            self.assertEqual(root, holder['server'].root / 'conversation' / 'receiver')
            self.assertIs(kwargs['server'], holder['server'])
            self.assertTrue(kwargs['playback_approved'])
            events.append('sink')
            return object()

        def run(directory, turns, **kwargs):
            events.append('run')
            self.assertTrue(holder['server'].client_ready.is_set())
            self.assertFalse(directory.exists())
            directory.mkdir()  # actual launcher's strict fresh-directory policy
            self.assertEqual(turns, ['fresh one', 'fresh two'])
            self.assertFalse(kwargs['idle_anchor'])
            self.assertTrue(kwargs['conversation_anchor'])
            self.assertEqual(kwargs['tts_mode'], 'complete')
            for name in ('capture_comet', 'allow_obs'):
                self.assertIs(kwargs[name], False)
            for name in ('asr_parent', 'obs_port', 'direct_audio_binary'):
                self.assertIsNone(kwargs[name])
            self.assertFalse(any('reference' in key for key in kwargs))
            kwargs['local_sink_factory'](directory / 'receiver', object())
            if mode == 'error':
                raise KeyboardInterrupt()
            if mode != 'noend':
                holder['server'].stream_state = 'ended'
                holder['server'].playback_done.set()
            return 17

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / 'fresh'
            expected = {'timeout': TimeoutError, 'cancel': RuntimeError,
                        'output': RuntimeError, 'error': KeyboardInterrupt,
                        'noend': RuntimeError}.get(mode)
            def invoke():
                return own(root, ['fresh one', 'fresh two'], server_factory=Server,
                           sink_factory=sink, run=run,
                           status=lambda: {'default_is_builtin_speaker': mode != 'output'},
                           clock=lambda: now[0], sleep=sleep,
                           emit=lambda text: events.append('print'))
            if expected:
                with self.assertRaises(expected):
                    invoke()
            else:
                self.assertEqual(invoke(), 17)
            receipt = json.loads((root / 'browser-owner.json').read_text())
            self.assertEqual(receipt['outcome'], 'ended' if mode == 'ok' else 'failed')
            self.assertNotIn('fresh one', json.dumps(receipt))
            self.assertEqual((root / 'browser-owner.json').stat().st_mode & 0o777, 0o600)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(ValueError):
                invoke()
        self.assertEqual(events[-1], 'closed')
        self.assertEqual(holder['server'].kwargs, dict(startup_seconds=1200, max_seconds=240))
        if mode in ('timeout', 'cancel', 'output'):
            self.assertNotIn('run', events)
        else:
            self.assertLess(events.index('print'), events.index('ready'))
            self.assertLess(events.index('ready'), events.index('run'))
            self.assertLess(events.index('run'), events.index('sink'))

    def test_positive_order_and_original_defaults(self):
        self.exercise()

    def test_no_gpu_before_readiness_timeout(self):
        self.exercise('timeout')

    def test_cancel(self):
        self.exercise('cancel')

    def test_output_fail(self):
        self.exercise('output')

    def test_interrupt_cleanup(self):
        self.exercise('error')

    def test_natural_end_required(self):
        self.exercise('noend')

    def test_offline_default(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            main([])
        self.assertEqual(json.loads(out.getvalue())['reference'], 'original')


if __name__ == '__main__':
    unittest.main()
