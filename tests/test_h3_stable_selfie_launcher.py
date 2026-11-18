"""Device-free stable selfie launcher contract tests."""
import ast
import contextlib
import io
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from research import launch_h3_direct_conversation as L


class StableSelfieLauncherTests(unittest.TestCase):
    def ready(self):
        from research.hf_h3_direct_conversation import endurance_limits
        windows, frames, _ = endurance_limits(False, False, False)
        return dict(type='h3_direct_ready', status='ready', session='test',
                    profile_sha256='a'*64, epoch=1, max_segments=windows,
                    max_total_frames=frames, port=7860, expires_at=time.time()+100,
                    visual_intents=True, attended_visual=True,
                    stable_selfie=True, anchor_policy='conversation-tail-every3-v1')

    def test_ready_accepts_exact_opt_in(self):
        row = self.ready()
        self.assertIs(L.validate_ready(row, 'test', visual_intents=True,
                       attended_visual=True, stable_selfie=True), row)

    def test_ready_rejects_missing_or_wrong_binding(self):
        for change in ({'stable_selfie': False}, {'stable_selfie': 1},
                       {'anchor_policy': 'epoch-idle-every3-v1'}, {'anchor_policy': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                L.validate_ready(dict(self.ready(), **change), 'test', visual_intents=True,
                                 attended_visual=True, stable_selfie=True)
        row = self.ready()
        del row['stable_selfie']
        with self.assertRaises(ValueError):
            L.validate_ready(row, 'test', visual_intents=True, attended_visual=True, stable_selfie=True)

    def test_ready_cannot_silently_adopt_static_mode(self):
        with self.assertRaises(ValueError):
            L.validate_ready(self.ready(), 'test', visual_intents=True, attended_visual=True)

    def test_legacy_ready_unchanged(self):
        row = self.ready()
        del row['stable_selfie']; del row['anchor_policy']
        self.assertIs(L.validate_ready(row, 'test', visual_intents=True, attended_visual=True), row)

    def test_ready_rejects_invalid_mode(self):
        for changes in ({'stable_selfie': 1}, {'attended_visual': False},
                        {'visual_intents': False}, {'turn_interruptions': True},
                        {'paired_cuts': True}):
            args = dict(stable_selfie=True, attended_visual=True, visual_intents=True)
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                L.validate_ready(self.ready(), 'test', **args)

    def test_run_rejects_invalid_combinations_before_any_selection(self):
        base = dict(stable_selfie=True, attended_visual=True, visual_intents=True,
                    conversation_anchor=True, tts_mode='native_voxcpm')
        for changes in ({'stable_selfie': 1}, {'attended_visual': False},
                        {'visual_intents': False}, {'conversation_anchor': False},
                        {'idle_anchor': True}, {'headless_interrupt': True},
                        {'review_av': True}, {'native_tts_cancel': True},
                        {'media_transport': 'webrtc-rtp-v1'}):
            args = dict(base, **changes)
            with self.subTest(changes=changes), patch.object(L, 'ordered_selection') as select:
                with self.assertRaises(ValueError):
                    L.run(Path('/tmp/not-created-stable-selfie'), ordered_references={}, **args)
                select.assert_not_called()

    def test_ready_accepts_device_free_static_without_attended(self):
        row = self.ready()
        del row['attended_visual']
        self.assertIs(L.validate_ready(row, 'test', visual_intents=True, stable_selfie=True), row)
        with self.assertRaises(ValueError):
            L.validate_ready(row, 'test', visual_intents=True)
        with self.assertRaises(ValueError):
            L.validate_ready(row, 'test', visual_intents=True, stable_selfie=True, paired_cuts=True)

    def test_typed_static_api_passes_early_guard_with_plain_review(self):
        for review in (False, True):
            with self.subTest(review=review), patch.object(L, 'ordered_selection', side_effect=LookupError('reached selection')) as select:
                with self.assertRaisesRegex(LookupError, 'reached selection'):
                    L.run(Path('/tmp/not-created-stable-selfie'), ['Fresh one', 'Fresh two'],
                          stable_selfie=True, visual_intents=True, conversation_anchor=True,
                          tts_mode='native_voxcpm', review_av=review, ordered_references={})
                select.assert_called_once()

    def test_typed_static_rejects_device_mix_before_selection(self):
        base = dict(stable_selfie=True, visual_intents=True, conversation_anchor=True,
                    tts_mode='native_voxcpm', review_av=True)
        for changes in ({'capture_comet':True}, {'asr_parent':Path('/tmp/parent')},
                        {'allow_obs':True}, {'obs_port':48124}, {'direct_audio_binary':Path('/bin/echo')},
                        {'local_sink_factory':lambda:None}, {'on_browser':lambda:None},
                        {'asr_rtc_factory':lambda:None}, {'start_gate':'/tmp/gate'},
                        {'long_session':True}, {'five_minute_test':True}, {'ten_minute_test':True}):
            with self.subTest(changes=changes), patch.object(L, 'ordered_selection') as select:
                with self.assertRaises(ValueError):
                    L.run(Path('/tmp/not-created-stable-selfie'), ['Fresh one', 'Fresh two'],
                          ordered_references={}, **dict(base, **changes))
                select.assert_not_called()
        for turns in (None, [], ['one'], ['one','two','three'], ['one',' ']):
            with self.subTest(turns=turns), patch.object(L, 'ordered_selection') as select:
                with self.assertRaises(ValueError):
                    L.run(Path('/tmp/not-created-stable-selfie'), turns, ordered_references={}, **base)
                select.assert_not_called()

    def test_cli_static_device_free_offline_and_mixes(self):
        argv=['--directory','/tmp/not-created-stable-selfie','--stable-selfie',
              '--visual-intents','--vox-parent','/tmp/parent','--review-av']
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            L.main(argv)
        self.assertIn('continuous static archive', stdout.getvalue())
        for extra in (['--capture-comet'], ['--allow-obs'], ['--obs-port','48124'],
                      ['--headless-interrupt'], ['--asr-parent','/tmp/asr']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                L.main(argv+extra)
            self.assertEqual(caught.exception.code,2)

    def test_plain_review_never_sets_native_paired_cut_binding(self):
        tree = ast.parse(Path(L.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        paired_assign = next(n for n in ast.walk(run) if isinstance(n,ast.Assign)
                             and ast.unparse(n.targets[0]) == "row['paired_cuts']")
        guard = next(n for n in ast.walk(run) if isinstance(n,ast.If) and paired_assign in n.body)
        self.assertEqual(ast.unparse(guard.test),'review_av and (not stable_selfie)')

    def ancestry_logs(self, count=4, export=True):
        rows=[]
        for target in (1,2):
            rows.append(dict(type='h3_visual_intent_received',session='a'*32,target_epoch=target,
                             intent_epoch=0,instruction_sha256=None,visual_achievement='unverified'))
        for index in range(count):
            row=dict(type='h3_visual_conditioning',session='a'*32,sequence=index,
                     target_epoch=min(index,2),intent_epoch=0,stage='native_conditioning_applied',
                     prompt_sha256='b'*64,visual_achievement='unverified')
            if export:
                row['visual_ancestry_selection']=dict(policy='conversation-tail-every3-v1',
                    sequence=index,kind='unconditioned' if index==0 else 'anchor' if index%3==0 else 'latest',
                    anchor_sha256=None if index==0 else 'c'*64,
                    selected_sha256=None if index==0 else 'c'*64 if index in (1,3) else 'd'*64,
                    anchor_sequence=None if index==0 else 1)
            rows.append(row)
        rows.append(dict(type='h3_native_tts_receipts',receipts=[dict(epoch=1),dict(epoch=2)]))
        return rows

    def collect(self, rows, directory, **kwargs):
        api=SimpleNamespace(fetch_job_logs=lambda **unused:[json.dumps(row) for row in rows])
        return L.collect_native_worker_receipts(api,SimpleNamespace(id='fixture'),directory,
            time.monotonic()+2,visual_intents=True,**kwargs)

    def test_collector_retains_actual_anchor_selection(self):
        with tempfile.TemporaryDirectory() as root:
            self.collect(self.ancestry_logs(),root,stable_selfie=True)
            evidence=json.loads((Path(root)/'visual-worker-evidence.json').read_text())
            self.assertEqual(evidence['windows'][3]['visual_ancestry_selection']['kind'],'anchor')
            self.assertEqual(evidence['windows'][3]['visual_ancestry_selection']['selected_sha256'], 'c'*64)

    def test_stable_null_targets_may_coalesce_without_movement(self):
        rows=self.ancestry_logs()
        for row in rows:
            if row.get('type')=='h3_visual_conditioning' and row['sequence']>0:
                row['target_epoch']=2
        with tempfile.TemporaryDirectory() as root:
            self.collect(rows,root,stable_selfie=True)
        for key,value in (('instruction_sha256','f'*64),('intent_epoch',1),('intent_epoch',False)):
            tampered=json.loads(json.dumps(rows));tampered[0][key]=value
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                self.collect(tampered,root,stable_selfie=True)
        missing=[r for r in rows if not (r.get('type')=='h3_visual_intent_received' and r['target_epoch']==1)]
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            self.collect(missing,root,stable_selfie=True)

    def test_collector_requires_export_and_anchor_exercise(self):
        for rows in (self.ancestry_logs(export=False),self.ancestry_logs(count=3)):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    self.collect(rows,root,stable_selfie=True)
                self.assertFalse((Path(root)/'visual-worker-evidence.json').exists())

    def test_collector_rejects_anchor_tampering(self):
        for sequence,key,value in ((2,'anchor_sha256','e'*64),(3,'selected_sha256','d'*64),
                                   (3,'kind','latest'),(1,'anchor_sequence',True)):
            rows=self.ancestry_logs()
            row=next(r for r in rows if r.get('type')=='h3_visual_conditioning' and r['sequence']==sequence)
            row['visual_ancestry_selection'][key]=value
            with self.subTest(sequence=sequence,key=key), tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                self.collect(rows,root,stable_selfie=True)

    def test_collector_rejects_unselected_export(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError,'unexpected'):
            self.collect(self.ancestry_logs(),root)
        with tempfile.TemporaryDirectory() as root:
            self.collect(self.ancestry_logs(export=False),root)

    def test_cli_rejects_missing_capability_without_side_effects(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            L.main(['--directory', '/tmp/not-created-stable-selfie', '--stable-selfie'])
        self.assertEqual(caught.exception.code, 2)

    def test_source_binds_session_and_forwards_stable_mode(self):
        tree = ast.parse(Path(L.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        assigns = [n for n in ast.walk(run) if isinstance(n, ast.Assign)]
        self.assertTrue(any(isinstance(n.targets[0], ast.Subscript)
                            and ast.unparse(n.targets[0]) == "row['stable_selfie']"
                            and isinstance(n.value, ast.Constant) and n.value.value is True
                            for n in assigns))
        stable_branch = next(n for n in ast.walk(run) if isinstance(n, ast.If)
                             and ast.unparse(n.test) == 'stable_selfie'
                             and any(isinstance(a, ast.Assign) and ast.unparse(a.targets[0]) == "row['stable_selfie']" for a in n.body))
        self.assertTrue(any(isinstance(a, ast.Assign)
                            and ast.unparse(a.targets[0]) == "row['turn_interruptions']"
                            and isinstance(a.value, ast.Constant) and a.value.value is False
                            for a in stable_branch.body))
        for name in ('await_ready', 'run_conversation', 'PiDirector', 'collect_native_worker_receipts'):
            call = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == name)
            self.assertTrue(any('stable_selfie' in ast.unparse(k.value) for k in call.keywords))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        attended_call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and n.func.id == 'run'
                             and any(k.arg == 'attended_visual' for k in n.keywords))
        anchor = next(k.value for k in attended_call.keywords if k.arg == 'conversation_anchor')
        self.assertEqual(ast.unparse(anchor), 'a.stable_selfie or not a.visual_intents')
        self.assertTrue(any(isinstance(n, ast.Assign)
                            and ast.unparse(n.targets[0]) == "direct_kwargs['stable_selfie']"
                            and isinstance(n.value, ast.Constant) and n.value.value is True
                            for n in ast.walk(main)))
        await_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'await_ready')
        validation = next(n for n in ast.walk(await_fn) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == 'validate_ready')
        self.assertTrue(any(k.arg == 'stable_selfie' and ast.unparse(k.value) == 'stable_selfie'
                            for k in validation.keywords))


if __name__ == '__main__':
    unittest.main()
