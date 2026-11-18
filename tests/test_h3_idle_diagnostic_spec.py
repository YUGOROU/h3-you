"""Offline admission/publication tests; no cloud or model execution."""
from dataclasses import asdict
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import sys
import types

from research.hf_h3_idle_diagnostic import SECTION, validate_spec, generated_files, publish
from research.h3_personal.fast_ref2va import config_for_layout
from research.launch_h3_fast_ref2va import IDENTITY_PREFIX, REFERENCE_SHA


def spec():
    return dict(idle_diagnostic=SECTION.copy(), fast_config=asdict(config_for_layout('portrait')),
                layout='portrait', attention_mode='fa4-packed', runtime_policy='resident-visual-cached',
                identity_prefix=IDENTITY_PREFIX, reference_sha256=REFERENCE_SHA,
                config={'timeout_seconds': 1200})


class AnchorEvidenceTests(unittest.TestCase):
    def fixtures(self, root):
        import json
        coverage = root.resolve()/'coverage'; coverage.mkdir()
        for i in range(14):
            guides = [None if i == 0 else ('a' if i in (1, 3, 6, 9, 12) else 'b')*64,
                      None if i == 0 else format(i, '064x'), format(i+20, '064x')]
            row = dict(idle_anchor_interval=3,
                       streaming=dict(sequence=i, logical_index=i, role='listening'),
                       committed_frame=124+i*102, canonical_sha256=guides[2],
                       guide_digests=guides, secret_text='must not escape')
            for rank in (0, 1):
                (coverage/f'streaming-rank-{rank}-window-{i}.json').write_text(json.dumps(row))
        return coverage

    def test_actual_proof_exclusive_and_sanitized(self):
        from research.hf_h3_idle_diagnostic import write_anchor_evidence
        with TemporaryDirectory() as tmp:
            root = Path(tmp); coverage = self.fixtures(root)
            proof = write_anchor_evidence(coverage, root)
            self.assertEqual(len(proof['windows']), 14)
            self.assertNotIn('secret_text', (root/'anchor-evidence.json').read_text())
            with self.assertRaises(FileExistsError): write_anchor_evidence(coverage, root)

    def retire_fixture(self, coverage):
        import hashlib
        archive = coverage/('retired-'+hashlib.sha256(b'idle-diagnostic').hexdigest()[:24])
        archive.mkdir()
        for receipt in coverage.glob('streaming-rank-*.json'):
            receipt.rename(archive/receipt.name)
        return archive

    def test_actual_retired_layout(self):
        from research.hf_h3_idle_diagnostic import write_anchor_evidence
        with TemporaryDirectory() as tmp:
            root = Path(tmp); coverage = self.fixtures(root)
            self.retire_fixture(coverage)
            proof = write_anchor_evidence(coverage, root)
            self.assertEqual(len(proof['windows']), 14)
            self.assertNotIn('secret_text', (root/'anchor-evidence.json').read_text())

    def test_ambiguous_mixed_and_symlink_layouts_rejected(self):
        import shutil
        from research.hf_h3_idle_diagnostic import write_anchor_evidence
        cases = ('two_archives', 'live_and_retired', 'mixed', 'partial_live',
                 'receipt_link', 'dangling_link', 'archive_link', 'coverage_link',
                 'invalid_archive', 'receipt_directory')
        for case in cases:
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root = Path(tmp); coverage = self.fixtures(root)
                archive = self.retire_fixture(coverage)
                receipt = archive/'streaming-rank-0-window-0.json'
                if case == 'two_archives':
                    shutil.copytree(archive, coverage/('retired-'+'f'*24))
                elif case == 'live_and_retired':
                    for path in archive.iterdir():
                        shutil.copyfile(path, coverage/path.name)
                elif case == 'mixed':
                    receipt.rename(coverage/receipt.name)
                elif case == 'partial_live':
                    shutil.copyfile(receipt, coverage/receipt.name)
                elif case in ('receipt_link', 'dangling_link'):
                    target = root/'target.json'
                    receipt.rename(target)
                    if case == 'dangling_link': target.unlink()
                    receipt.symlink_to(target)
                elif case == 'archive_link':
                    target = root/'archive'; archive.rename(target)
                    archive.symlink_to(target, target_is_directory=True)
                elif case == 'coverage_link':
                    target = root/'real-coverage'; coverage.rename(target)
                    coverage.symlink_to(target, target_is_directory=True)
                elif case == 'invalid_archive':
                    archive.rename(coverage/'retired-not-a-digest')
                elif case == 'receipt_directory':
                    receipt.unlink(); receipt.mkdir()
                with self.assertRaises(ValueError):
                    write_anchor_evidence(coverage, root)
                self.assertFalse((root/'anchor-evidence.json').exists())

    def test_invalid_actual_receipts_fail_without_output(self):
        import json
        from research.hf_h3_idle_diagnostic import write_anchor_evidence
        for key, value in [('idle_anchor_interval', True), ('idle_anchor_interval', 4),
                           ('guide_digests', ['f'*64]*3),
                           ('canonical_sha256', 'invalid'), ('committed_frame', 0)]:
            with self.subTest(key=key, value=value), TemporaryDirectory() as tmp:
                root = Path(tmp); coverage = self.fixtures(root)
                path = coverage/'streaming-rank-1-window-3.json'
                row = json.loads(path.read_text()); row[key] = value
                path.write_text(json.dumps(row))
                with self.assertRaises(ValueError): write_anchor_evidence(coverage, root)
                self.assertFalse((root/'anchor-evidence.json').exists())

    def test_both_ranks_wrong_visual_anchor_rejected(self):
        import json
        from research.hf_h3_idle_diagnostic import write_anchor_evidence
        with TemporaryDirectory() as tmp:
            root = Path(tmp); coverage = self.fixtures(root)
            for rank in (0, 1):
                path = coverage/f'streaming-rank-{rank}-window-6.json'
                row = json.loads(path.read_text()); row['guide_digests'][0] = 'f'*64
                path.write_text(json.dumps(row))
            with self.assertRaises(ValueError): write_anchor_evidence(coverage, root)


class SpecTests(unittest.TestCase):
    def test_accepted(self):
        self.assertEqual(validate_spec(spec()), SECTION)

    def test_anchor_exact_variant(self):
        from research.hf_h3_idle_diagnostic import ANCHOR_SECTION
        row = spec(); row['idle_diagnostic'] = ANCHOR_SECTION.copy()
        self.assertEqual(validate_spec(row), ANCHOR_SECTION)
        for value in ('other', None, True):
            row['idle_diagnostic']['anchor_policy'] = value
            with self.assertRaises(ValueError): validate_spec(row)
        row['idle_diagnostic'] = dict(ANCHOR_SECTION, extra=True)
        with self.assertRaises(ValueError): validate_spec(row)
        self.assertNotIn('anchor_policy', SECTION)

    def test_worker_anchor_environment_restored(self):
        import os
        import time
        from research.hf_h3_idle_diagnostic import run_session, ANCHOR_SECTION, ANCHOR_ENV, ANCHOR_POLICY
        runtime = Mock()
        def start():
            self.assertEqual(os.environ[ANCHOR_ENV], ANCHOR_POLICY)
            raise RuntimeError('start failed')
        runtime.start.side_effect = start
        api = Mock(); api.repo_info.return_value.private = True
        row = spec(); row['idle_diagnostic'] = ANCHOR_SECTION.copy()
        with patch.dict(os.environ, {'HF_TOKEN': 'test', ANCHOR_ENV: 'previous'}), patch.dict(sys.modules, {
                'huggingface_hub': types.SimpleNamespace(HfApi=lambda **kw: api)}), patch(
                'research.h3_personal.accepted_profiles.choose_profile', return_value=types.SimpleNamespace(name='reference1024')), patch(
                'research.h3_personal.ref2va_runtime.sha', return_value='0'*64), patch(
                'research.h3_personal.live_voice_runtime.LiveVoiceRuntime', return_value=runtime), patch(
                'research.hf_h3_idle_diagnostic.publish') as publication:
            with self.assertRaises(RuntimeError):
                run_session('/tmp/root', dict(reference='/tmp/ref', personal='/tmp/p', base='/tmp/b', smd='/tmp/s'), row, 'p/r', time.monotonic()+1000)
            self.assertEqual(os.environ[ANCHOR_ENV], 'previous')
            runtime.close.assert_called_once()
            self.assertEqual(publication.call_args.kwargs['anchor_policy'], ANCHOR_POLICY)
            with self.assertRaises(ValueError):
                run_session('/tmp/root', {}, spec(), 'p/r', time.monotonic()+1000)

    def test_exclusive_even_null(self):
        for key in ('locked_speech', 'streaming_trial', 'conversation_scenario',
                    'live_voice_session', 'live_direct_session', 'live_relay_session',
                    'continuation_diagnostic', 'late_listening_diagnostic'):
            with self.subTest(key=key):
                row = spec(); row[key] = None
                with self.assertRaises(ValueError): validate_spec(row)

    def test_exact_section(self):
        for section in (None, {}, dict(SECTION, windows=True), dict(SECTION, windows=14.),
                        dict(SECTION, extra=True), dict(SECTION, windows=13)):
            row = spec(); row['idle_diagnostic'] = section
            with self.assertRaises(ValueError): validate_spec(row)

    def test_runtime_constraints(self):
        for key, value in (('layout', 'landscape'), ('runtime_policy', 'baseline'),
                           ('attention_mode', 'fa2'), ('reference_sha256', '0'*64),
                           ('face_crop_box', [0, 0, 1, 1]), ('regional_compile', True),
                           ('config', {'timeout_seconds': 1200.})):
            row = spec(); row[key] = value
            with self.assertRaises(ValueError): validate_spec(row)
        row = spec(); key = next(iter(row['fast_config'])); row['fast_config'][key] = None
        with self.assertRaises(ValueError): validate_spec(row)

    def test_explicit_generated_only_names(self):
        report = dict(windows=14, native_samples=42, encoded_samples=42, final=True)
        names = generated_files(report)
        self.assertEqual(len(names), 129)
        self.assertIn('native-manifest.json', names)
        self.assertIn('segments/native-013.mkv', names)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn('segments/segment-013.mp4', names)
        self.assertIn('13-106-native.png', names)
        self.assertFalse(any('latent' in n or 'driver' in n for n in names))
        with self.assertRaises(ValueError): generated_files(dict(report, final=False))

    def test_partial_publication_is_explicit_private_and_sanitized(self):
        class Add:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        api = Mock()
        api.repo_info.return_value.private = True
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                'huggingface_hub': types.SimpleNamespace(CommitOperationAdd=Add)}):
            root = Path(tmp)
            (root/'segments').mkdir()
            for name in ('segments/native-000.mkv', 'native-manifest.json',
                         'samples.json', 'driver.json', 'latents.pt', 'raw.wav'):
                (root/name).write_bytes(b'generated')
            publish(api, 'private/repo', root, {'files': ['raw.wav'], 'error': '/secret'}, partial=True)
            ops = api.create_commit.call_args.kwargs['operations']
            self.assertEqual({o.path_in_repo for o in ops}, {
                'idle-diagnostic/failure.json', 'idle-diagnostic/segments/native-000.mkv',
                'idle-diagnostic/native-manifest.json', 'idle-diagnostic/samples.json'})
            self.assertNotIn(b'/secret', ops[0].path_or_fileobj)
            api.repo_info.return_value.private = False
            with self.assertRaises(ValueError): publish(api, 'public/repo', root, {}, partial=True)

    def test_publication_rejects_symlinks_and_total_overflow(self):
        class Add:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        api = Mock(); api.repo_info.return_value.private = True
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                'huggingface_hub': types.SimpleNamespace(CommitOperationAdd=Add)}):
            root = Path(tmp)
            (root/'secret').write_bytes(b'private')
            (root/'samples.json').symlink_to(root/'secret')
            with self.assertRaises(ValueError): publish(api, 'p/r', root, {}, partial=True)
            (root/'samples.json').unlink()
            (root/'samples.json').write_bytes(b'x'*100)
            with patch('research.hf_h3_idle_diagnostic.LIMIT', 100):
                with self.assertRaises(ValueError): publish(api, 'p/r', root, {}, partial=True)
            api.create_commit.assert_not_called()

    def test_manifest_cannot_widen_whitelist(self):
        report = dict(windows=14, native_samples=42, encoded_samples=42, final=True)
        report['files'] = generated_files(report) + ['driver/private.json']
        with self.assertRaises(ValueError): generated_files(report)
        self.assertNotIn('driver/private.json', generated_files(report, partial=True))

    def test_bootstrap_admission_precedes_tts_and_assets(self):
        from pathlib import Path
        text = Path('research/bootstrap_h3_fast_ref2va.py').read_text()
        admission = text.index("if 'idle_diagnostic' in spec:")
        self.assertLess(admission, text.index('tts_future=None'))
        self.assertLess(admission, text.index("for asset in json.loads"))
        self.assertIn('from research.hf_h3_idle_diagnostic import run_session', text)


if __name__ == '__main__':
    unittest.main()
