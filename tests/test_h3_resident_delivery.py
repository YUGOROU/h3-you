"""Offline contracts: no network, subprocess or devices."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from research.h3_personal.resident_delivery import ResidentDelivery, request, timing


class ResidentDeliveryTests(unittest.TestCase):
    def result(self):
        return dict(origin='comet_process_audio_endpoint_asr', terminal_state='running',
                    resident_lease_id='owned', deadline=time.time()+100,
                    asr=dict(completion_policy='input-finalized', input_committed=True,
                             final=True, capture_stopped=True, cleanup_complete=True,
                             on_device=False, errors=[], text='こんにちは',
                             events=[dict(final=True, text='こんにちは', endpoint_source='native_completed')],
                             audio_sent_seconds=2, finalized_audio_seconds=2),
                    proposal=dict(model='Qwen/Qwen3.8-27B:cerebras', reply='はい'))

    def validate(self, r):
        return request(r, 'session', 2, 'owned', r['deadline'], time.time())

    def test_running_not_mutated(self):
        r = self.result()
        original = copy.deepcopy(r)
        row = self.validate(r)
        self.assertEqual(row['sequence'], 2)
        self.assertEqual(r, original)

    def test_fail_closed(self):
        for key, value in [('terminal_state', 'paused'), ('resident_lease_id', 'other'),
                           ('origin', 'fake'), ('deadline', 0)]:
            r = self.result()
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(r)

    def test_full_native_required(self):
        for key, value in [('input_committed', False), ('capture_stopped', False),
                           ('cleanup_complete', False), ('on_device', True),
                           ('finalized_audio_seconds', 1), ('audio_sent_seconds', float('nan'))]:
            r = self.result()
            r['asr'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(r)

    def test_longer_asr_does_not_change_spoken_reply_limit(self):
        r=self.result();r['asr']['text']='あ'*200
        r['asr']['events'][0]['text']=r['asr']['text']
        self.assertEqual(self.validate(r)['reply'],'はい')
        r['asr']['text']='あ'*501;r['asr']['events'][0]['text']=r['asr']['text']
        with self.assertRaises(ValueError):self.validate(r)

    def test_exact_model(self):
        r = self.result()
        r['proposal']['model'] = 'fallback'
        with self.assertRaises(ValueError):
            self.validate(r)

    def test_asr_and_h3_have_independent_finite_deadlines(self):
        r = self.result()
        now = time.time()
        row = request(r, 'session', 1, 'owned', now+60, now, r['deadline'])
        self.assertLessEqual(row['expires_at'], now+60)
        with self.assertRaises(ValueError):
            request(r, 'session', 1, 'owned', now+60, now, r['deadline']+1)

    def test_gap_lead(self):
        self.assertEqual(timing(10, 13)['gap_seconds'], 3)
        self.assertEqual(timing(10, 8)['lead_seconds'], 2)
        self.assertEqual(timing(10, 10)['gap_seconds'], 0)
        self.assertIsNone(timing(None, 10)['gap_seconds'])
        with self.assertRaises(ValueError):
            timing(10, float('inf'))


class ResidentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'deployment').mkdir()
        self.deadline = time.time() + 100
        (self.root / 'submission.json').write_text(json.dumps(dict(
            job_id='job', resident_lease_id='owned', deadline=self.deadline)))
        (self.root / 'deployment/spec.json').write_text(json.dumps(dict(
            asr_policy='endpoint-native-v1', config=dict(timeout_seconds=1200),
            publication_repo='YUGOROU/private', live_voice_session=dict(id='session'))))
        obs = self.root / 'obs.json'
        obs.write_text(json.dumps(dict(host='127.0.0.1', port=12345,
                                      audio_monitor_device='BlackHole2ch_UID', pid=123)))
        self.files = {'live/state.json': json.dumps(dict(
            status='ready', session_id='session', previous_audio_reused=False,
            expires_at=self.deadline)).encode()}
        self.api = Mock()
        self.api.repo_info.return_value = SimpleNamespace(
            private=True, sha='ready', last_modified=SimpleNamespace(timestamp=lambda: 0))
        self.api.list_repo_files.side_effect = lambda *a, **kw: list(self.files)
        self.api.inspect_job.return_value.status.stage = 'RUNNING'
        self.fetch = Mock(side_effect=self.fetch_file)
        self.delivery = ResidentDelivery(self.root, obs, api=self.api, fetch=self.fetch,
                                        commit_add=lambda **kw: kw)
        (self.root / 'capture-metadata.jsonl').write_text(json.dumps(dict(
            type='stopped', cleanup=True, frames=10, nonzero_samples=10)))
        self.result = ResidentDeliveryTests.result(self)
        self.result['deadline'] = self.deadline
        self.api.create_commit.side_effect = self.commit
        self.probe = patch('research.h3_personal.resident_delivery.subprocess.run',
                           return_value=SimpleNamespace(stdout=json.dumps(dict(
                               format=dict(duration='2'), streams=[])))).start()
        self.addCleanup(patch.stopall)
        self.codecs = patch('research.h3_personal.resident_delivery.verify_codecs').start()

    def fetch_file(self, name, revision):
        self.assertIn(revision, ('ready', 'output'))
        path = self.root / 'remote' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        return path

    def commit(self, *args, **kwargs):
        operation = kwargs['operations'][0]
        self.files[operation['path_in_repo']] = operation['path_or_fileobj'].getvalue()

    def submit(self):
        return self.delivery.submit(self.result, self.root, 1)

    def publish(self, pending):
        self.api.repo_info.return_value.sha = 'output'
        name = 'live/turn-001/review.mp4'
        self.files[name] = b'verified AV bytes'
        self.files['live/turn-001/result.json'] = json.dumps(dict(
            schema='h3-live-voice-state-v1', status='output', session_id='session',
            sequence=1, previous_audio_reused=False,
            reply_sha256=hashlib.sha256(pending['row']['reply'].encode()).hexdigest(),
            files={name: hashlib.sha256(self.files[name]).hexdigest()})).encode()

    def test_ambiguous_commit_ack_recovers_without_resubmission(self):
        def ambiguous(*args, **kwargs):
            self.commit(*args, **kwargs)
            self.assertIn(1, self.delivery.turns)
            self.assertIsNotNone(self.delivery.pending)
            raise ConnectionError('ack lost')
        self.api.create_commit.side_effect = ambiguous
        with self.assertRaises(ConnectionError):
            self.submit()
        pending = self.delivery.recover_pending(1)
        with self.assertRaises(ValueError):
            self.submit()
        with self.assertRaises(ValueError):
            self.delivery.recover_pending(2)
        changed = self.delivery.recover_pending(1)
        changed['row']['reply'] = 'mutated'
        self.assertEqual(self.delivery.recover_pending(1), pending)
        self.publish(pending)
        self.assertTrue(self.delivery.finish(pending).exists())
        self.assertEqual(self.api.create_commit.call_count, 1)
        with self.assertRaises(ValueError):
            self.delivery.recover_pending(1)
        self.fetch.assert_any_call('live/input-001.json', 'output')

    def test_finish_retry_after_probe_failure_reuses_identical_file(self):
        pending = self.submit()
        self.publish(pending)
        success = self.probe.return_value
        self.probe.side_effect = [subprocess.CalledProcessError(1, 'ffprobe'), success]
        with self.assertRaises(subprocess.CalledProcessError):
            self.delivery.finish(pending)
        path = self.root / 'resident-turn-001/reply.mp4'
        before = path.stat().st_mtime_ns
        self.assertEqual(self.delivery.recover_pending(1), pending)
        self.assertEqual(self.delivery.finish(pending), path)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(self.api.create_commit.call_count, 1)

    def test_finish_receipt_failure_keeps_pending_for_same_asset_retry(self):
        pending = self.submit()
        self.publish(pending)
        with patch.object(self.delivery, '_save', side_effect=OSError('receipt unavailable')):
            with self.assertRaises(OSError):self.delivery.finish(pending)
        self.assertEqual(self.delivery.recover_pending(1), pending)
        self.assertTrue(self.delivery.finish(pending).exists())
        self.assertEqual(self.api.create_commit.call_count, 1)

    def test_finish_transport_failure_keeps_pending(self):
        pending = self.submit()
        self.publish(pending)
        self.fetch.side_effect = ConnectionError('download failed')
        with self.assertRaises(ConnectionError):
            self.delivery.finish(pending)
        self.fetch.side_effect = self.fetch_file
        self.assertTrue(self.delivery.finish(self.delivery.recover_pending(1)).exists())
        self.assertEqual(self.api.create_commit.call_count, 1)

    def test_input_mismatch_rejected_even_with_matching_reply(self):
        pending = self.submit()
        self.publish(pending)
        row = dict(pending['row'], asr_sha256='different input')
        self.files['live/input-001.json'] = json.dumps(row).encode()
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            self.delivery.finish(pending)
        self.probe.assert_not_called()
        self.assertEqual(self.delivery.recover_pending(1), pending)

    def test_remote_hash_mismatch_rejected(self):
        pending = self.submit()
        self.publish(pending)
        self.files['live/turn-001/review.mp4'] = b'corrupt'
        with self.assertRaisesRegex(ValueError, 'AV hash mismatch'):
            self.delivery.finish(pending)
        self.probe.assert_not_called()

    def test_existing_different_file_never_overwritten(self):
        pending = self.submit()
        self.publish(pending)
        turn = self.root / 'resident-turn-001'
        turn.mkdir()
        path = turn / 'reply.mp4'
        path.write_bytes(b'other bytes')
        with self.assertRaisesRegex(ValueError, 'local AV hash mismatch'):
            self.delivery.finish(pending)
        self.assertEqual(path.read_bytes(), b'other bytes')
        self.probe.assert_not_called()

    def test_recovery_never_resets_original_wait_end(self):
        with patch('research.h3_personal.resident_delivery.time.monotonic', return_value=10):
            pending = self.submit()
        self.publish(pending)
        end = pending['end']
        with patch('research.h3_personal.resident_delivery.time.monotonic', return_value=end+1):
            recovered = self.delivery.recover_pending(1)
            self.assertEqual(recovered['end'], end)
            with self.assertRaises(TimeoutError):
                self.delivery.finish(recovered)
        self.assertEqual(self.api.create_commit.call_count, 1)
        self.probe.assert_not_called()

    def test_recover_asset_verifies_hash_and_never_starts_playback(self):
        pending = self.submit()
        self.publish(pending)
        path = self.delivery.finish(pending)
        self.probe.reset_mock()
        self.assertEqual(self.delivery.recover_asset(1), path)
        self.assertNotIn('play_start', self.delivery.turns[1])
        self.probe.assert_not_called()
        path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.delivery.recover_asset(1)

    def check_playback(self, success):
        pending = self.submit()
        self.publish(pending)
        path = self.delivery.finish(pending)
        self.probe.side_effect = None if success else subprocess.CalledProcessError(1, 'ffmpeg')
        with patch('research.h3_personal.resident_delivery.subprocess.check_output', return_value='123\n'):
            if success:
                self.delivery.play(path, 1)
            else:
                with self.assertRaises(subprocess.CalledProcessError):
                    self.delivery.play(path, 1)
            self.assertIs(self.delivery.turns[1]['play_success'], success)
            self.assertIn('play_end', self.delivery.turns[1])
            calls = self.probe.call_count
            with self.assertRaises(ValueError):
                self.delivery.recover_asset(1)
            with self.assertRaises(ValueError):
                self.delivery.play(path, 1)
            self.assertEqual(self.probe.call_count, calls)

    def test_failed_playback_not_successful_or_replayable(self):
        self.check_playback(False)

    def test_successful_playback_not_replayable(self):
        self.check_playback(True)


if __name__ == '__main__':
    unittest.main()
