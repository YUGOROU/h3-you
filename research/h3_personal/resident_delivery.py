"""Resident native AV delivery. Never owns a GPU or changes device routing.

submit uploads synchronously (no playback wait); finish polls independently of play.
The session coordinator owns cancellation and may run play on its own thread.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from io import BytesIO

from research.deliver_h3_endpoint_turn import verify_codecs
from research.h3_personal.endpoint_asr import native_completed_text, finalized_covers
from research.hf_h3_live_voice import validate_request


def timing(previous_play_end, next_asset_ready):
    """Positive gap means late asset; positive lead means asset ready early."""
    if previous_play_end is None:
        return dict(previous_play_end=None, next_asset_ready=next_asset_ready,
                    gap_seconds=None, lead_seconds=None)
    if not all(isinstance(x, (int, float)) and math.isfinite(x)
               for x in (previous_play_end, next_asset_ready)):
        raise ValueError('finite timestamps required')
    delta = next_asset_ready - previous_play_end
    return dict(previous_play_end=previous_play_end, next_asset_ready=next_asset_ready,
                gap_seconds=max(0, delta), lead_seconds=max(0, -delta))


def request(result, session, sequence, lease, deadline, now, asr_deadline=None):
    asr_deadline = deadline if asr_deadline is None else asr_deadline
    a = result['asr']
    finals = [e for e in a.get('events', []) if e.get('final') is True]
    if (result.get('origin') != 'comet_process_audio_endpoint_asr' or
        result.get('terminal_state') != 'running' or result.get('error_type') or
        result.get('resident_lease_id') != lease or result.get('deadline') != asr_deadline or
        now >= min(deadline, asr_deadline) or a.get('completion_policy') != 'input-finalized' or
        a.get('input_committed') is not True or a.get('final') is not True or
        a.get('capture_stopped') is not True or a.get('cleanup_complete') is not True or
        a.get('on_device') is not False or a.get('errors') or not finals or
        any(e.get('endpoint_source') != 'native_completed' for e in finals) or
        native_completed_text(finals) != a.get('text') or
        not isinstance(a.get('text'), str) or not 1 <= len(a['text']) <= 500):
        raise ValueError('clean resident native whole-input final required')
    sent, done = a.get('audio_sent_seconds'), a.get('finalized_audio_seconds')
    if (type(sent) not in (int, float) or type(done) not in (int, float) or
        not math.isfinite(sent) or not math.isfinite(done) or sent <= 0 or
        not finalized_covers(done, sent)):
        raise ValueError('whole input coverage required')
    row = dict(schema='h3-live-voice-input-v1', session_id=session, sequence=sequence,
               origin=result['origin'], asr_final=True, asr_stopped=True,
               asr_on_device=False, endpoint_source='native_completed',
               director_model=result['proposal']['model'], reply=result['proposal']['reply'],
               asr_sha256=hashlib.sha256(a['text'].encode()).hexdigest(),
               expires_at=min(now + 45, deadline))
    return validate_request(row, session, sequence, 'endpoint-native-v1')


class ResidentDelivery:
    def __init__(self, directory, obs_receipt, *, api=None, fetch=None, commit_add=None):
        self.directory = Path(directory).resolve()
        self.submission_path = self.directory / 'submission.json'
        self.submission_bytes = self.submission_path.read_bytes()
        submission = json.loads(self.submission_bytes)
        spec = json.loads((self.directory / 'deployment/spec.json').read_text())
        if (submission.get('provider', 'huggingface_jobs') != 'huggingface_jobs' or
            not submission.get('job_id') or spec.get('asr_policy') != 'endpoint-native-v1' or
            spec['config']['timeout_seconds'] not in (1200,1500)):
            raise ValueError('owned native1200/1500 session required')
        self.job = submission['job_id']
        self.repo = spec['publication_repo']
        self.session = spec['live_voice_session']['id']
        self.lease = submission['resident_lease_id']
        self.timeout = spec['config']['timeout_seconds']
        self.deadline = submission['deadline']
        self.asr_deadline = submission.get('resident_asr_deadline', self.deadline)
        if not self.lease or not self.repo.startswith('YUGOROU/'):
            raise ValueError('owned lease/private repo required')
        self.obs = json.loads(Path(obs_receipt).read_text())
        if (self.obs.get('host') != '127.0.0.1' or
            type(self.obs.get('port')) is not int or not 1024 <= self.obs['port'] <= 65535 or
            self.obs.get('audio_monitor_device') != 'BlackHole2ch_UID'):
            raise ValueError('existing OBS route required')
        if api is None:
            from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
            from research.provision_modal_identity_secret import cached_token
            token = cached_token()
            api = HfApi(token=token)
            fetch = lambda name, revision: Path(hf_hub_download(self.repo, name,
                repo_type='dataset', revision=revision, token=token))
            commit_add = CommitOperationAdd
        self.api, self.fetch, self.commit_add = api, fetch, commit_add
        self.lock = threading.RLock()
        self.play_lock = threading.Lock()
        self.pending = None
        self.closed = False
        self.turns = {}
        info, names = self._snapshot()
        ready = json.loads(self.fetch('live/state.json', info.sha).read_text())
        if (ready.get('status') != 'ready' or ready.get('session_id') != self.session or
            ready.get('previous_audio_reused') is not False or
            ready.get('expires_at') != self.deadline or
            not 0 < self.deadline - time.time() <= self.timeout or
            any(n.startswith(('live/input-', 'live/turn-')) for n in names)):
            raise ValueError('fresh pinned READY required')
        self.ready_revision = info.sha
        self.ready_at = info.last_modified.timestamp()
        self._alive()

    def _alive(self):
        if self.closed or time.time() >= self.deadline:
            raise RuntimeError('resident lease closed')
        if self.submission_path.read_bytes() != self.submission_bytes:
            raise ValueError('session owner changed')
        if self.api.inspect_job(job_id=self.job, namespace='YUGOROU').status.stage != 'RUNNING':
            raise RuntimeError('resident job not running')

    def _snapshot(self):
        info = self.api.repo_info(self.repo, repo_type='dataset')
        if not info.private or not info.sha:
            raise ValueError('private pinned revision required')
        return info, self.api.list_repo_files(self.repo, repo_type='dataset', revision=info.sha)

    def _save(self):
        rows = []
        for seq, row in sorted(self.turns.items()):
            row = dict(row)
            row.update(timing(self.turns.get(seq-1, {}).get('play_end'), row.get('asset_ready'))
                       if row.get('asset_ready') is not None else {})
            rows.append(row)
        path = self.directory / 'resident-delivery.json'
        temporary = path.with_suffix('.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(session_id=self.session, turns=rows, realtime_verified=False), f)
        os.replace(temporary, path)

    def submit(self, result, capture_dir, sequence):
        with self.lock:
            self._alive()
            if type(sequence) is not int or sequence != len(self.turns)+1 or sequence > 4 or self.pending:
                raise ValueError('ordered max4/maxpending1 required')
            capture = Path(capture_dir) / 'capture-metadata.jsonl'
            diagnostics = [json.loads(x) for x in capture.read_text().splitlines()]
            if (capture.stat().st_mtime <= self.ready_at or not diagnostics or
                diagnostics[-1].get('type') != 'stopped' or
                diagnostics[-1].get('cleanup') is not True or
                diagnostics[-1].get('frames', 0) <= 0 or diagnostics[-1].get('nonzero_samples', 0) <= 0):
                raise ValueError('fresh stopped capture required')
            row = request(result, self.session, sequence, self.lease, self.deadline, time.time(), self.asr_deadline)
            info, names = self._snapshot()
            prefix = f'live/turn-{sequence:03d}/'
            name = f'live/input-{sequence:03d}.json'
            if name in names or any(n.startswith(prefix) for n in names):
                raise ValueError('replay forbidden')
            # Retain ownership before upload: a transport error can hide a committed input.
            self.pending = dict(sequence=sequence, row=row, revision=info.sha,
                                end=time.monotonic()+min(120, self.deadline-time.time()))
            self.turns[sequence] = dict(sequence=sequence, submitted=time.time())
            self._save()
            self.api.create_commit(self.repo, repo_type='dataset', parent_commit=info.sha,
                operations=[self.commit_add(path_in_repo=name, path_or_fileobj=BytesIO(json.dumps(row).encode()))],
                commit_message=f'Fresh resident native sequence {sequence}')
            return copy.deepcopy(self.pending)

    def recover_pending(self, sequence):
        """Recover only this session's existing attempt; never upload or renew its wait."""
        with self.lock:
            self._alive()
            if (type(sequence) is not int or self.pending is None or
                self.pending['sequence'] != sequence):
                raise ValueError('unknown pending turn')
            return copy.deepcopy(self.pending)

    def finish(self, pending):
        with self.lock:
            if pending != self.pending or pending is None:
                raise ValueError('unknown pending turn')
            pending = copy.deepcopy(self.pending)
        seq = pending['sequence']
        prefix = f'live/turn-{seq:03d}/'
        while time.monotonic() < pending['end'] and time.time() < self.deadline:
            self._alive()
            info, names = self._snapshot()
            if prefix+'result.json' in names:
                name = f'live/input-{seq:03d}.json'
                if (name not in names or
                    json.loads(self.fetch(name, info.sha).read_text()) != pending['row']):
                    raise ValueError('published input mismatch')
                output = json.loads(self.fetch(prefix+'result.json', info.sha).read_text())
                if (info.sha == pending['revision'] or
                    output.get('schema') != 'h3-live-voice-state-v1' or output.get('status') != 'output' or
                    output.get('session_id') != self.session or output.get('sequence') != seq or
                    output.get('previous_audio_reused') is not False or
                    output.get('reply_sha256') != hashlib.sha256(pending['row']['reply'].encode()).hexdigest()):
                    raise ValueError('fresh bound output required')
                data = self.fetch(prefix+'review.mp4', info.sha).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if output['files'][prefix+'review.mp4'] != digest:
                    raise ValueError('AV hash mismatch')
                turn = self.directory / f'resident-turn-{seq:03d}'
                turn.mkdir(mode=0o700, exist_ok=True)
                media = turn / 'reply.mp4'
                try:
                    with media.open('xb') as f:
                        f.write(data)
                except FileExistsError:
                    if media.read_bytes() != data:
                        raise ValueError('local AV hash mismatch')
                media.chmod(0o600)
                probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                    '-of', 'json', str(media)], check=True, capture_output=True, text=True,
                    timeout=max(.01, min(10, self.deadline-time.time())))
                metadata = json.loads(probe.stdout)
                verify_codecs(metadata)
                duration = float(metadata['format']['duration'])
                if not math.isfinite(duration) or not 0 < duration <= 120:
                    raise ValueError('bounded media duration required')
                with self.lock:
                    self.turns[seq].update(path=str(media), sha256=digest, duration=duration,
                        output_revision=info.sha, manifest=output, asset_ready=time.time())
                    self._save()
                    self.pending = None
                return media
            time.sleep(min(.5, max(0, pending['end']-time.monotonic()), max(0, self.deadline-time.time())))
        raise TimeoutError('bounded resident output wait')

    def _verified_asset(self, sequence, path=None):
        self._alive()
        if type(sequence) is not int or sequence not in self.turns:
            raise ValueError('unknown asset turn')
        row = self.turns[sequence]
        saved = row.get('path')
        if (not saved or (path is not None and str(Path(path)) != saved) or
            'play_start' in row or
            (sequence > 1 and 'play_end' not in self.turns[sequence-1]) or
            hashlib.sha256(Path(saved).read_bytes()).hexdigest() != row['sha256']):
            raise ValueError('ordered verified asset required')
        return Path(saved)

    def recover_asset(self, sequence):
        """Return a verified unstarted asset, without starting playback."""
        with self.lock:
            return self._verified_asset(sequence)

    def play(self, path, sequence):
        with self.play_lock:
            with self.lock:
                self._verified_asset(sequence, path)
                row = self.turns[sequence]
                if str(self.obs['pid']) not in subprocess.check_output(['pgrep', '-x', 'OBS'], text=True, timeout=5).split():
                    raise ValueError('existing OBS missing')
                row['play_start'] = time.time()
                row['play_success'] = False
                self._save()
            try:
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-re', '-i', str(path),
                    '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb',
                    '-c:a', 'aac', '-b:a', '128k', '-f', 'mpegts', '-mpegts_flags', '+resend_headers',
                    '-muxdelay', '0', f"udp://127.0.0.1:{self.obs['port']}?pkt_size=1316"],
                    check=True, timeout=max(.01, min(120, self.deadline-time.time(), row['duration']+10)))
                with self.lock:
                    row['play_success'] = True
            finally:
                with self.lock:
                    row['play_end'] = time.time()
                    self._save()

    def close(self):
        # No job cancellation: only the session owner can terminate the lease.
        with self.lock:
            self.closed = True
