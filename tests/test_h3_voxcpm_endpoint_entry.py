import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

from research.endpoints.voxcpm_l4 import entrypoint as entry
from research.h3_personal.voxcpm_endpoint_protocol import PINS
from research.h3_personal.voxcpm2_streaming_candidate import MODEL_ID, MODEL_REVISION


class EntryTests(unittest.TestCase):
    def test_native_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'request.json').write_text(json.dumps(dict(candidate='voxcpm2', expires_at=1100,
                model=dict(repo=MODEL_ID, revision=MODEL_REVISION),
                reference=dict(sha256=PINS['reference_sha256']))))
            cuda = Mock()
            cuda.device_count.return_value = 1
            cuda.get_device_name.return_value = 'NVIDIA L4'
            cuda.get_device_properties.return_value.total_memory = 24000
            torch = types.SimpleNamespace(cuda=cuda, __version__='pinned', version=types.SimpleNamespace(cuda='12'))
            server = Mock()
            candidate = types.SimpleNamespace(sample_rate=48000)
            with patch.object(entry.sys, 'argv', ['entry', directory, '--native']), \
                 patch.object(entry.sys, 'path', list(entry.sys.path)), \
                 patch.object(entry.os, 'chdir'), patch.object(entry.os, 'umask'), \
                 patch.dict(entry.os.environ, H3_TTS_KEY='k'*32, H3_TTS_SESSION='a'*32, H3_TTS_BOOT='b'*32), \
                 patch.object(entry.time, 'time', return_value=1000), \
                 patch.object(entry.signal, 'signal'), patch.object(entry.signal, 'setitimer') as alarm, \
                 patch.dict('sys.modules', torch=torch), \
                 patch('research.hf_h3_native_tts_probe.models', return_value=('model','ref')) as models, \
                 patch('research.h3_personal.voxcpm2_streaming_candidate.load', return_value=candidate) as load, \
                 patch('research.h3_personal.voxcpm_endpoint_service.create_server', return_value=server) as create, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                entry.main()
            alarm.assert_called_once_with(entry.signal.ITIMER_REAL, 100)
            load.assert_called_once_with('model', 'ref', optimize=False)
            self.assertEqual(create.call_args.kwargs['host'], '0.0.0.0')
            server.serve_forever.assert_called_once()
            server.server_close.assert_called_once()
            self.assertEqual(json.loads(output.getvalue())['device_count'], 1)

    def test_completion_only_full_drain(self):
        candidate = types.SimpleNamespace(sample_rate=48000,
            completion=dict(reason='natural_stop', natural_stop=True, stream_exhausted=True),
            stream=lambda text: iter([[.25, -.25]]))
        wrapper = entry.LoggedCandidate(candidate, lambda: 7)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            list(wrapper.stream('private text'))
        row = json.loads(output.getvalue())
        self.assertEqual(row['epoch'], 7)
        self.assertEqual(row['samples'], 2)
        self.assertTrue(row['completion']['natural_stop'])
        self.assertNotIn('private', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            stream = wrapper.stream('private')
            next(stream)
            stream.close()
        self.assertEqual(output.getvalue(), '')

    def test_cancel_completion_logs_prefix_and_prior_reason_not_natural_eos(self):
        class Candidate:
            sample_rate=48000
            completion=None
            def stream(self,text):
                self.completion=dict(reason='unknown',natural_stop=False,stream_exhausted=False)
                try:
                    yield [.25]
                    yield [.5]
                except GeneratorExit:
                    self.completion.update(reason='cancelled',cancel_prior_reason='unknown')
                    raise
        wrapper=entry.LoggedCandidate(Candidate(),lambda:1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            stream=wrapper.stream('private direction')
            next(stream);stream.close()
        row=json.loads(output.getvalue())
        self.assertEqual(row['samples'],1)
        self.assertEqual(row['completion']['reason'],'cancelled')
        self.assertEqual(row['completion']['cancel_prior_reason'],'unknown')
        self.assertIs(row['completion']['natural_stop'],False)
        self.assertEqual(wrapper.export_evidence()['completions'],[row])
        self.assertNotIn('private direction',output.getvalue())

    def test_bootstrap_target_preserves_legacy(self):
        tree = ast.parse(Path('research/hf_h3_native_tts_probe.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bootstrap')
        self.assertEqual(fn.args.args[-1].arg, 'reexec_target')
        self.assertIsNone(fn.args.defaults[-1].value)
        self.assertIn("reexec_target if reexec_target is not None", ast.unparse(fn))


if __name__ == '__main__':
    unittest.main()
