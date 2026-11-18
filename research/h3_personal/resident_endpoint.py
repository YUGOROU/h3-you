"""Finite attended ASR session. No Pi, raw-media persistence, or implicit resume."""
import copy
import json
import math
from decimal import Decimal
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import uuid

from research.run_h3_asr_endpoint import assert_deployment, control, deployment, pause_owned, reserve, sha
from research.run_h3_endpoint_voice import stop_capture, validate_capture_time, wait_for_capture
from research.h3_personal.endpoint_asr import transcribe, finalized_covers, _host
from research.h3_personal.zoom_input import CaptureInterrupted, InputBlocked
from research.h3_personal.private_root import root as _private_root

PRIVATE = _private_root()
SOURCE = Path(__file__).resolve().parents[1]
# Consecutive all-zero capture windows before the input path is declared dead.
# One silent window is a normal pause; three is ~30s of a source that never
# produced a single nonzero sample.
SILENT_INPUT_WINDOWS = 3


class TurnRejected(ValueError):
    """Input validation rejects this turn without ending a warm lease."""


class NoSpeech(TurnRejected):
    """A clean silent turn is not a session or in-flight generation failure."""


class InputPathSilent(RuntimeError):
    """The capture source delivered no audible sample across consecutive windows.

    Distinct from NoSpeech: a human who stays silent for one window is normal, but
    a source that yields all-zero PCM for several windows is a broken input path
    (wrong app/tap target, muted tab, output routed elsewhere). Treating that as
    an ordinary silent turn lets a long session spin indefinitely without ever
    producing a turn, so it must stop the conversation instead.
    """


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _parent(original, campaign, info):
    expected = json.loads((campaign / 'intent.json').read_text())
    json.loads((campaign / 'source-pins.json').read_text())
    result = campaign / 'result.json'
    if result.exists():
        receipt = json.loads(result.read_text())
    else:
        receipt = json.loads((campaign / 'external-cleanup.json').read_text())
        if receipt.get('original_result_exists') is not False:
            raise ValueError('explicit missing-result cleanup required')
    if receipt.get('terminal_state') != 'paused':
        raise ValueError('parent must be paused')
    transport=expected['model']['env'].get('H3_ASR_MEDIA_TRANSPORT')
    if transport not in (None,'webrtc-exact-v1'):raise ValueError('parent ASR transport')
    marker = expected['model']['env'].get('H3_ASR_SESSION_SECONDS')
    if marker not in (None, '10800'):
        raise ValueError('parent ASR session duration marker')
    reviewed = deployment(original['model']['repository'], expected['model']['revision'],
                          original['name'], int(expected['model']['env']['H3_ASR_EXPIRES_AT']),
                          rtc=transport=='webrtc-exact-v1',
                          session_seconds=10800 if marker == '10800' else 1800)
    assert_deployment(expected, reviewed)
    if not info.private or info.sha != expected['model']['revision']:
        raise ValueError('private reviewed branch required')
    return expected


def _accounting_receipt(accounting):
    if accounting not in ('legacy-reservation', 'operator-managed'):
        raise ValueError('accounting must be legacy-reservation or operator-managed')
    return dict(accounting=accounting,
                reservation_usd='.75' if accounting == 'legacy-reservation' else None,
                reservation=None)


class ResidentEndpoint:
    """Use ``with ResidentEndpoint.open(directory, parent_campaign) as lease``."""

    @classmethod
    def open(cls, directory: Path, parent_campaign: Path, *, accounting='legacy-reservation', rtc_factory=None, coowned_vox_directory=None, session_seconds=1800):
        """Open a finite lease; explicit 3h sessions require operator-managed accounting."""
        if type(session_seconds) is not int or session_seconds not in (1800, 10800):
            raise ValueError('ASR session_seconds must be1800 or explicit10800')
        if session_seconds == 10800 and accounting != 'operator-managed':
            raise ValueError('long ASR session requires operator-managed accounting')
        if rtc_factory is not None and (getattr(rtc_factory,'media_transport',None)!='webrtc-exact-v1'
                or not callable(getattr(rtc_factory,'open_turn',None))):raise ValueError('explicit exact RTC ASR factory required')
        if rtc_factory is not None and accounting!='operator-managed':raise ValueError('RTC ASR uses operator-managed accounting')
        accounting_receipt = _accounting_receipt(accounting)
        from huggingface_hub import HfApi
        from research.provision_modal_identity_secret import cached_token
        from research.probe_h3_asr_cpu_space import http_ready

        root = directory.resolve()
        if root.exists() or not root.is_relative_to(PRIVATE):
            raise ValueError('fresh private campaign required')
        self = cls()
        self.root, self.closed, self.owned = root, False, False
        self._rtc_factory=rtc_factory;self._rtc_epoch=0
        # Consecutive capture windows whose source PCM was entirely zero. Reset by
        # any window that carries a nonzero sample, so ordinary pauses never trip it.
        self._silent_windows = 0
        self.lease_id = 'hf-asr-resident-' + uuid.uuid4().hex
        self._key = secrets.token_urlsafe(40)
        self._token = cached_token()
        self._intent = None
        self.receipt = dict(resident_lease_id=self.lease_id, terminal_state='unconfirmed',
                            raw_audio_saved=False, max_seconds=session_seconds,
                            max_replicas=1 if rtc_factory is not None else 3, **accounting_receipt)
        root.mkdir(mode=0o700)
        try:
            api = HfApi(token=self._token)
            if api.whoami()['name'] != 'YUGOROU':
                raise ValueError('owned namespace required')
            original = json.loads((PRIVATE / 'hf-asr-endpoint-20261019-v1/intent.json').read_text())
            self.name = original['name']
            self._path = '/v2/endpoint/YUGOROU/' + self.name
            current = control(self._token, self._path)
            if current['status']['state'] != 'paused':
                raise ValueError('current actual paused state required')
            previous = _parent(original, parent_campaign, api.repo_info(original['model']['repository'], repo_type='model'))
            assert_deployment(current, previous)
            allowed_peer = None
            if coowned_vox_directory is not None:
                from research.h3_personal.voxcpm_https_lease import GUARD, NAME
                peer_root = Path(coowned_vox_directory).resolve()
                guard = json.loads(GUARD.read_text())
                if (peer_root != root.parent.parent/'vox'
                        or guard.get('directory') != str(peer_root)
                        or guard.get('name') != NAME
                        or guard.get('namespace') != 'YUGOROU'
                        or guard.get('expires_at', 0) <= time.time()):
                    raise ValueError('same campaign live Vox owner required')
                allowed_peer = NAME
            if any(e['status']['state'] not in ('paused', 'scaledToZero')
                   and e['name'] != allowed_peer
                   for e in control(self._token, '/v2/endpoint/YUGOROU')['items']):
                raise ValueError('another allocation exists')
            providers = control(self._token, '/v2/provider/YUGOROU')
            rows = [c for v in providers['vendors'] if v['name'] == 'aws'
                    for r in v['regions'] if r['name'] == 'us-east-1'
                    for c in r['computes'] if c['instanceType'] == 'nvidia-t4' and c['instanceSize'] == 'x1']
            if len(rows) != 1 or rows[0]['status'] != 'available' or Decimal(str(rows[0]['pricePerHour'])) != Decimal('.5') or rows[0]['quota']['maxAccelerators'] - rows[0]['quota']['usedAccelerators'] < (1 if rtc_factory is not None else 3):
                raise ValueError('fresh rate/quota gate failed')
            self._binary = root / 'comet-pcm'
            capture_source = SOURCE / 'stream_h3_comet_pcm.swift'
            subprocess.run(['swiftc', '-swift-version', '6', '-strict-concurrency=complete', '-warnings-as-errors', str(capture_source), '-o', str(self._binary)], check=True, timeout=60, capture_output=True)
            stage = root / 'deployment'
            stage.mkdir(mode=0o700)
            if rtc_factory is not None:
                from research.prepare_h3_asr_rtc import stage as stage_rtc
                stage_rtc(stage)
            else:
                for name in ('gate.py', 'entrypoint.py'):
                    shutil.copyfile(SOURCE / 'endpoints/nemotron_t4' / name, stage / name)
            self._pins = {n: sha(stage / n) for n in ('gate.py', 'entrypoint.py')}
            self._pins.update(capture_source=sha(capture_source), capture_binary=sha(self._binary))
            _write(root / 'source-pins.json', self._pins)
            commit = api.upload_folder(repo_id=original['model']['repository'], repo_type='model', folder_path=stage, commit_message='Fresh finite resident ASR lease; no media or secrets')
            self.deadline = int(time.time()) + session_seconds
            self._intent = deployment(original['model']['repository'], commit.oid, self.name,
                                      self.deadline, rtc=rtc_factory is not None,
                                      session_seconds=session_seconds)
            _write(root / 'intent.json', self._intent)
            self._reserve_accounting()
            self.receipt.update(name=self.name, session_deadline=self.deadline)
            patch = copy.deepcopy(self._intent)
            patch['model']['secrets'] = {'H3_ASR_KEY': self._key}
            self.owned = True  # PUT can succeed despite a lost response.
            control(self._token, self._path, dict(model=patch['model'], tags=patch['tags'],
                compute=patch['compute']), 'PUT')
            if control(self._token, self._path)['status']['state'] == 'paused':
                control(self._token, self._path + '/resume', {}, 'POST')
            end = time.monotonic() + 300
            while True:
                current = control(self._token, self._path)
                if time.monotonic() >= end or time.time() >= self.deadline - 75:
                    raise RuntimeError('finite readiness expired')
                if current['status']['state'] == 'running':
                    break
                if current['status']['state'] == 'failed':
                    raise RuntimeError('readiness failed')
                time.sleep(5)
            assert_deployment(current, self._intent)
            self._url = current['status']['url']
            host = _host(self._url)
            anon, _ = http_ready(host)
            code, body = http_ready(host, self._token)
            if (anon not in (401, 403, 404) or code != 200 or json.loads(body).get('device') != 'cuda'
                    or rtc_factory is not None and json.loads(body).get('media_transport')!='webrtc-exact-v1'
                    or time.monotonic() >= end):
                raise ValueError('authenticated CUDA/anonymous denial required within readiness bound')
            return self
        except BaseException as exc:
            self.receipt['error_type'] = type(exc).__name__
            self.close()
            raise

    def _reserve_accounting(self):
        if self.receipt['accounting'] == 'legacy-reservation':
            reserve(PRIVATE / 'compute-budget.json', self.lease_id,
                    scope='hf-endpoint-ASR-T4-attended-comet')
            self.receipt['reservation'] = self.lease_id

    @property
    def intent(self):
        return copy.deepcopy(self._intent)

    def capture(self, directory: Path, planned_start: float | None, *, capture_seconds=45,
                on_asr_event=None, input_pipeline=None) -> dict:
        if on_asr_event is not None and not callable(on_asr_event):
            raise ValueError('on_asr_event must be callable')
        import threading
        process = None
        capture_lock = threading.Lock()
        capture_cancelled = threading.Event()
        def stop_input():
            with capture_lock:
                capture_cancelled.set()
                stop_capture(process)
        try:
            if self.closed or self._key is None:
                raise RuntimeError('lease closed or cleanup pending')
            validate_capture_time(planned_start)
            if type(capture_seconds) is not int or not 8 <= capture_seconds <= 45:
                raise ValueError('finite8..45 second input window required')
            if ((time.time() if planned_start is None else planned_start) > self.deadline - 75):
                raise ValueError('capture requires 75 seconds of lease remaining')
            root = directory.resolve()
            if root.exists() or not root.is_relative_to(self.root):
                raise ValueError('fresh lease-local turn directory required')
            root.mkdir(mode=0o700)
            current = control(self._token, self._path)
            assert_deployment(current, self._intent)
            if current['status']['state'] != 'running':
                raise RuntimeError('resident endpoint not running; never resume')
            if input_pipeline is not None:
                if input_pipeline.source_contract['capture_source_sha256'] != self._pins['capture_source']:
                    raise ValueError('Zoom capture source pin drift')
                input_pipeline.prepare_capture(deadline=time.monotonic() + min(65, self.deadline-time.time()-75))
                planned_start = time.time() + 3
                print(f'Approved Zoom capture starts in 3 seconds ({capture_seconds} seconds).', flush=True)
            timing = wait_for_capture(planned_start, self.deadline)
            if sha(self._binary) != self._pins['capture_binary']:
                raise ValueError('capture binary drift')
            with (root / 'capture-metadata.jsonl').open('xb') as metadata:
                from research.h3_personal.local_vad import VADBoundary
                boundary = None
                def start():
                    nonlocal process, boundary
                    if time.time() > self.deadline - 75:
                        raise RuntimeError('capture deadline exceeded')
                    argv = ['--stream'] if input_pipeline is None else input_pipeline.capture_argv
                    if input_pipeline is not None and argv != ['--stream', '--app', 'zoom']:
                        raise ValueError('explicit Zoom stream argv required')
                    with capture_lock:
                        if capture_cancelled.is_set():
                            raise CaptureInterrupted('capture already cancelled')
                        process = subprocess.Popen([str(self._binary), *argv], stdout=subprocess.PIPE, stderr=metadata)
                        boundary = (VADBoundary(process.stdout) if input_pipeline is None else
                                    input_pipeline.wrap_capture(process.stdout, capture_seconds=capture_seconds))
                    return boundary
                rtc_turn=None;transcript=None
                try:
                    factory=getattr(self,'_rtc_factory',None)
                    if factory is not None:
                        self._rtc_epoch+=1  # Reserve before uncertain allocation; never replay.
                        rtc_turn=factory.open_turn(endpoint=self._url,token=self._token,key=self._key,
                            session=self.lease_id,epoch=self._rtc_epoch,expires_at=self.deadline,
                            deadline=time.monotonic()+min(90,self.deadline-time.time()))
                    transcript = transcribe(self._url, token=self._token, owner_key=self._key, pcm=start,
                                            timeout=60, max_audio_seconds=capture_seconds, completion_policy='input-finalized',
                                            on_input_stop=stop_input,
                                            **({'_transport':rtc_turn.transport} if rtc_turn is not None else {}),
                                            **({} if on_asr_event is None else {'on_event': on_asr_event}))
                finally:
                    try:
                        try:stop_input()
                        finally:
                            if process is not None:process.stdout.close()
                    finally:
                        if rtc_turn is not None:
                            native_done=bool(transcript and transcript.get('cleanup_complete') and transcript.get('input_committed')
                                and transcript.get('errors') in ([],['no_native_final']))
                            rtc_turn.close(require_complete=native_done)
                            if transcript is not None:
                                transcript['media_transport']='webrtc-exact-v1'
                                transcript['rtc_carrier']=rtc_turn.receipt
            diagnostics = [json.loads(line) for line in (root / 'capture-metadata.jsonl').read_text().splitlines()]
            last = diagnostics[-1] if diagnostics else {}
            if input_pipeline is None:
                transcript.update(input_boundary_policy=VADBoundary.policy,
                                  input_boundary_reason=None if boundary is None else boundary.reason,
                                  vad=None if boundary is None else boundary.evidence())
            else:
                transcript['zoom_capture'] = input_pipeline.finish_capture(last, transcript, capture_seconds=capture_seconds)
                transcript['input_source'] = input_pipeline.source_contract.copy()
            if process is None or last.get('type') != 'stopped' or last.get('cleanup') is not True or last.get('frames', 0) <= 0:
                raise ValueError('stopped capture cleanup receipt required')
            # Any window carrying a nonzero sample proves the source is alive, so
            # a later silent window starts counting from zero again.
            if last.get('nonzero_samples', 0) != 0:
                self._silent_windows = 0
            if (input_pipeline is None and boundary is not None
                    and boundary.speech_start is None and transcript['cleanup_complete']
                    and transcript['audio_sent_seconds'] == 0
                    and transcript['input_committed'] is False
                    and transcript['errors'] == ['no_input_audio']):
                # Distinguish "the human was quiet" from "the source is dead".
                # Only the actual capture receipt decides; VAD reason alone cannot.
                if last.get('nonzero_samples', 0) == 0:
                    self._silent_windows += 1
                if self._silent_windows >= SILENT_INPUT_WINDOWS:
                    self.receipt['last_rejected_turn_code'] = 'input_path_silent'
                    self.receipt['silent_windows'] = self._silent_windows
                    _write(root / 'result.json', dict(transcript, timing=timing,
                        skipped='input_path_silent', silent_windows=self._silent_windows))
                    raise InputPathSilent(
                        'capture source delivered all-zero PCM for '
                        f'{self._silent_windows} consecutive windows; input path is not audible')
                _write(root / 'result.json', dict(transcript, timing=timing, skipped='no_speech'))
                raise NoSpeech('local VAD rejected window; no ASR turn committed')
            if (last.get('nonzero_samples', 0) == 0 and transcript['cleanup_complete']
                    and transcript['input_committed'] and transcript['errors'] == ['no_native_final']
                    and finalized_covers(transcript['finalized_audio_seconds'], transcript['audio_sent_seconds'])):
                _write(root / 'result.json', dict(transcript, timing=timing, skipped='no_speech'))
                raise NoSpeech('clean silent turn')
            _write(root / 'asr-diagnostics.json', dict(transcript, timing=timing))
            if not transcript['final'] or transcript['errors'] or not transcript['cleanup_complete'] or not transcript['input_committed'] or transcript['completion_policy'] != 'input-finalized' or transcript['audio_sent_seconds'] <= 0 or not finalized_covers(transcript['finalized_audio_seconds'], transcript['audio_sent_seconds']):
                raise TurnRejected('full_native_finalized_coverage_required')
            if not 1 <= len(transcript['text'].strip()) <= 500:
                raise TurnRejected('input_exceeds_500_characters')
            transcript.update(capture_stopped=True, terminal_state='running', resident_lease_id=self.lease_id, session_deadline=self.deadline)
            _write(root / 'result.json', dict(transcript, timing=timing))
            return transcript
        except CaptureInterrupted:
            # Transient capture-infrastructure loss: reject the turn, keep the
            # healthy lease; the conversation may retry within its own budget.
            self.receipt['last_rejected_turn_code'] = 'zoom_capture_interrupted'
            raise
        except InputBlocked:
            self.receipt['last_rejected_turn_code'] = 'zoom_input_not_eligible'
            raise  # Operator decline/cancel does not fault a healthy ASR lease.
        except TurnRejected as exc:
            self.receipt['last_rejected_turn_code'] = str(exc)
            raise
        except InputPathSilent:
            # The ASR lease is healthy; the capture source is not. The receipt and
            # result.json are already written, so keep the lease rather than
            # tearing down a warm Endpoint for a local input-path fault.
            raise
        except BaseException as exc:
            self.receipt['error_type'] = type(exc).__name__
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        try:
            if self.owned:
                self.receipt['terminal_state'] = pause_owned(self._token, self._path)
                if self.receipt['terminal_state'] != 'paused':
                    raise RuntimeError('pause not confirmed')
        finally:
            self._key = None
            target = self.root / 'result.json'
            temporary = self.root / 'result.tmp'
            temporary.write_text(json.dumps(self.receipt, ensure_ascii=False, indent=2))
            temporary.chmod(0o600)
            temporary.replace(target)
            self.closed = not self.owned or self.receipt.get('terminal_state') == 'paused'

    def __enter__(self):
        if self.closed:
            raise RuntimeError('lease closed')
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.receipt['error_type'] = kind.__name__
        self.close()
        return False
