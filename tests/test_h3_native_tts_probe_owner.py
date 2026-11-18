import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from research.run_h3_native_tts_probe import bundle,bootstrap,main,FILES,ROOT

class ProbeOwnerTests(unittest.TestCase):
    def test_trace_abba_same_candidate_seed_and_warmup_cleanup(self):
        from research.hf_h3_native_tts_probe import trace_abba, validate_profile
        from types import SimpleNamespace
        from unittest.mock import Mock
        for spec in ({'candidate':'voxcpm2','voxcpm_trace_abba':True},
                     {'candidate':'qwen3','voxcpm_trace_abba':True}):
            with self.assertRaises(ValueError): validate_profile(spec)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--candidate','voxcpm2','--voxcpm-trace-abba'])
        candidate=SimpleNamespace(observe_stop=True)
        torch=Mock()
        torch.cuda.max_memory_allocated.return_value=1
        torch.cuda.max_memory_reserved.return_value=2
        calls=[]
        def measure(c,text,path,kind):
            calls.append((c,text,path,c.observe_stop))
            (path/'native.wav').parent.mkdir(parents=True,exist_ok=True)
            (path/'native.wav').touch()
            return dict(iterator_completion_seconds=1,duration_seconds=2,
                        formal_completion_accepted=c.observe_stop)
        with tempfile.TemporaryDirectory() as d:
            warm,rows=trace_abba(candidate,'greeting',Path(d),'float',torch,measure)
            self.assertEqual([x[3] for x in calls],[False,True,False,True,True,False])
            self.assertTrue(all(c is candidate and t=='greeting' for c,t,_,_ in calls))
            self.assertTrue(all(not p.exists() for _,_,p,_ in calls[:2]))
            self.assertEqual([r['formal_completion_accepted'] for r in rows],[False,True,True,False])
            self.assertEqual(len(warm),2)
            self.assertTrue(all(r['seed']==42 and r['matched_text'] for r in warm+rows))
        self.assertEqual(torch.manual_seed.call_count,6)
        self.assertEqual(torch.cuda.manual_seed_all.call_count,6)
        torch.manual_seed.assert_called_with(42)
        torch.cuda.manual_seed_all.assert_called_with(42)

    def test_optimize_profile_offline_and_qwen_rejection(self):
        from research.hf_h3_native_tts_probe import validate_profile
        self.assertFalse(validate_profile({'candidate':'voxcpm2'}))
        self.assertTrue(validate_profile({'candidate':'voxcpm2','voxcpm_optimize':True}))
        for value in (True, 'false', 1):
            with self.assertRaises(ValueError):
                validate_profile({'candidate':'qwen3','voxcpm_optimize':value})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--candidate','qwen3','--voxcpm-optimize'])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--candidate','voxcpm2','--voxcpm-optimize']),0)
        self.assertTrue(json.loads(output.getvalue())['voxcpm_optimize'])

    def test_compiler_is_compile_only_and_records_version(self):
        from research.hf_h3_native_tts_probe import prepare_compiler
        from types import SimpleNamespace
        with patch('research.hf_h3_native_tts_probe.subprocess.run') as run:
            prepare_compiler({'candidate':'voxcpm2'})
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as d, patch('research.hf_h3_native_tts_probe.shutil.which',return_value='/usr/bin/gcc'), patch('research.hf_h3_native_tts_probe.subprocess.run',return_value=SimpleNamespace(stdout='gcc test\n')):
            with patch.dict('os.environ',{},clear=False):
                prepare_compiler({'candidate':'voxcpm2','voxcpm_optimize':True,'_compiler_receipt_root':d})
            self.assertEqual(json.loads((Path(d)/'compiler.json').read_text())['version'],'gcc test')

    def test_supply_diagnostic_profile_is_eager_only(self):
        from research.run_h3_native_tts_probe import validate_diagnostic
        from research.hf_h3_native_tts_probe import validate_profile
        validate_diagnostic('voxcpm2',False,False,True)
        validate_diagnostic('voxcpm2',False,False,False,True,True)
        self.assertFalse(validate_profile({'candidate':'voxcpm2','voxcpm_stage_diagnostic':True,'voxcpm_lm_compile':True}))
        with self.assertRaises(ValueError):validate_diagnostic('voxcpm2',False,False,False,False,True)
        with self.assertRaises(ValueError):validate_profile({'candidate':'voxcpm2','voxcpm_lm_compile':True})
        self.assertFalse(validate_profile({'candidate':'voxcpm2','voxcpm_supply_diagnostic':True}))
        for candidate,optimize,abba in [('qwen3',False,False),('voxcpm2',True,False),('voxcpm2',True,True)]:
            with self.assertRaises(ValueError):validate_diagnostic(candidate,optimize,abba,True)
            with self.assertRaises(ValueError):validate_profile({'candidate':candidate,'voxcpm_optimize':optimize,'voxcpm_trace_abba':abba,'voxcpm_supply_diagnostic':True})

    def test_offline_no_network(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('network')):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main([]),0)
    def test_bundle_only_allowlisted_sources_no_voice_env(self):
        with tempfile.TemporaryDirectory() as d:
            path=bundle(Path(d),dict(candidate='voxcpm2',reference={'sha256':'a'*64}))
            with zipfile.ZipFile(path) as z:
                self.assertEqual(set(z.namelist()),set(FILES)|{'research/__init__.py','research/h3_personal/__init__.py','request.json'})
                self.assertFalse(any(n.endswith('.wav') or '.venv' in n or '.env' in n for n in z.namelist()))
                self.assertEqual(z.read('research/__init__.py'),b'')
                self.assertEqual(json.loads(z.read('request.json'))['candidate'],'voxcpm2')
    def test_bootstrap_is_valid_python_no_credential_literal(self):
        code=bootstrap('YUGOROU/probe','a'*40,'b'*64)
        compile(code,'bootstrap','exec')
        self.assertIn("u.hostname=='huggingface.co' or u.hostname.endswith('.hf.co')",code)
        self.assertIn("if host=='huggingface.co' else {}",code)
        self.assertNotIn('hf_secret',code)
        self.assertIn("'..' not in p.parts",code)
