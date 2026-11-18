"""Composable finite Mac typed-text controller; no deployment or device setup.

Caller supplies an already opened authenticated consumer SocketAdapter and fresh
RAM-only typed turns. No CLI: endpoint attestation/READY acquisition belongs to
its owner, not an implicit cloud launcher. Ownership transfers for this run.
"""
from collections import deque
import json
import re
from pathlib import Path
import select
import shutil
import subprocess
import threading
import time
import uuid

from research.h3_personal.director_runtime import MODEL, child_environment
from research.h3_personal.live_relay_commands import Commands
from research.h3_personal.paired_av_pump import ReceiverPump
from research.h3_personal.headless_av_sink import HeadlessAVSink
from research.h3_personal.obs_av_sink import OBSAVSink
from research.h3_personal.local_av_diagnostics import persist


def direct_audio_pin(binary, delay=0.2):
    """Inspect only: never execute the local output binary."""
    import hashlib
    import math
    import os
    if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 1:
        raise ValueError('audio delay must be finite within 0..1')
    path = Path(binary).expanduser()
    if path.is_symlink():
        raise ValueError('direct audio binary must not be a symlink')
    path = path.resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('regular executable direct audio binary required')
    return dict(audio_binary=str(path), binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                audio_delay_seconds=float(delay))


def _persist_audio_evidence(root, sink):
    """Scalar encoder-input receipt only; no transcript/audio or route claims."""
    import math
    import os
    evidence=getattr(sink,'audio_evidence',None)
    if evidence is None:
        return
    counts=('source_pcm_samples','padding_pcm_samples','source_pcm16_samples',
            'nonzero_pcm16_samples','peak_pcm16','sum_squares_pcm16','ack_count')
    if (type(evidence) is not dict or any(type(evidence.get(k)) is not int or
            evidence[k]<0 for k in counts) or
            type(evidence.get('rms_pcm16')) not in (int,float) or
            not math.isfinite(evidence['rms_pcm16']) or evidence['rms_pcm16']<0 or
            evidence.get('encoder_input_only') is not True or
            evidence.get('remote_verified') is not False):
        raise ValueError('scalar sink audio evidence required')
    clean={k:evidence[k] for k in (*counts,'rms_pcm16')}
    for key in ('device_queue_only', 'obs_audio_disabled',
                'device_playback_completed', 'schedule_alignment_only'):
        if key in evidence:
            if type(evidence[key]) is not bool:
                raise ValueError('boolean route evidence required')
            clean[key] = evidence[key]
    clean.update(encoder_input_only=True,remote_verified=False)
    path=Path(root)/'encoder-audio-evidence.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as out:
        json.dump(clean,out,sort_keys=True,allow_nan=False)
        out.write('\n')


def _persist_output_timing(root, sink):
    """Local queue scheduling only: no audio content or remote acceptance claim."""
    from collections.abc import Mapping
    import math
    import os

    timing = getattr(sink, 'timing', None)
    from research.h3_personal.local_av_diagnostics import timing_evidence, persist
    # Legacy sinks without diagnostic timing keys produce no receipt at all.
    if not isinstance(timing, Mapping) or not any(
            key in timing for key in ('video_epoch', 'first_av_submission_monotonic',
                                      'accepted_monotonic', 'events')):
        return
    if 'video_epoch' not in timing:
        persist(root, 'unified-av-timing.json', timing_evidence(timing))
        return
    floats = ('video_epoch', 'audio_epoch', 'audio_delay_seconds',
              'first_av_submission_monotonic', 'desired_first_video_write',
              'actual_first_video_write', 'first_video_write_skew',
              'max_video_lateness')
    counts = ('queue_highwater', 'max_decoded_outstanding',
              'video_frames_written', 'ack_count')
    clean = {}
    for key in floats:
        value = timing[key]
        if value is not None and (type(value) not in (int, float) or
                                  not math.isfinite(value)):
            raise ValueError('finite timing scalar required')
        clean[key] = None if value is None else float(value)
    for key in counts:
        value = timing[key]
        if type(value) is not int or value < 0:
            raise ValueError('nonnegative timing count required')
        clean[key] = value
    for key in ('schedule_alignment_only', 'remote_verified'):
        if type(timing[key]) is not bool:
            raise ValueError('boolean timing evidence required')
    if (timing['schedule_alignment_only'] is not True or
            timing['remote_verified'] is not False or
            timing['ack_scope'] != 'local_raw_queue_and_device_audio_queue'):
        raise ValueError('local queue scheduling evidence required')
    clean.update(schedule_alignment_only=True, remote_verified=False,
                 ack_scope='local_raw_queue_and_device_audio_queue')
    if 'accepted_monotonic' in timing:
        values = timing['accepted_monotonic']
        if (type(values) not in (tuple, list) or len(values) > 144 or
                any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in values)):
            raise ValueError('bounded finite queue timestamps required')
        clean['accepted_monotonic'] = [float(v) for v in values]
    path = Path(root) / 'split-av-timing.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(clean, out, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    # The sanitized allowlist receipt is written only after all strict validation.
    persist(root, 'unified-av-timing.json', timing_evidence(timing))


def _receiver_diagnostic(exc):
    """Copy bounded structural metadata only; never format exceptions or locals."""
    def token(value):
        return ''.join(c if c.isascii() and (c.isalnum() or c in '._-') else '_'
                       for c in value[:80]) or 'unknown'

    frames = deque(maxlen=8)
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        filename = code.co_filename.replace('\\', '/').rsplit('/', 1)[-1]
        frames.append(f'{token(filename)}:{token(code.co_name)}:'
                      f'{min(max(tb.tb_lineno, 0), 999999999)}')
        tb = tb.tb_next
    return f'{token(type(exc).__name__)} [{", ".join(frames)}]'


def proposal_reply(row, *, visual_intents=False, stable_selfie=False):
    """Extract agent.mjs's flattened validateDraft proposal, never assistant prose."""
    if type(visual_intents) is not bool:raise ValueError('explicit visual proposal contract required')
    if type(stable_selfie) is not bool or (stable_selfie and not visual_intents):
        raise ValueError('stable selfie requires the visual proposal contract')
    if visual_intents:
        from research.h3_personal.conversation_conditioning import validate_visual_intent
        fields={'type','turn_id','epoch','model','baseline','schema','reply','visual_intent',
                'status','effects_executed','renderer_attached','audio_attached'}
        if type(row) is not dict or set(row)!=fields or row.get('schema')!='h3-director-visual-turn-v1':
            raise ValueError('versioned visual proposal required')
        intent = validate_visual_intent(row.get('visual_intent'))
        if stable_selfie and intent['instruction'] is not None:
            raise ValueError('stable selfie forbids camera/action instructions')
    elif type(row) is dict and ('visual_intent' in row or 'schema' in row):
        raise ValueError('visual proposal cannot enter a legacy reply-only route')
    if (type(row) is not dict or row.get('type') != 'turn'
            or row.get('model') != MODEL or row.get('status') != 'proposal_only'
            or row.get('effects_executed') is not False
            or row.get('renderer_attached') is not False
            or row.get('audio_attached') is not False
            or type(row.get('turn_id')) is not str or not row['turn_id']
            or type(row.get('epoch')) is not int or row['epoch'] < 1
            or (not visual_intents and (
                row.get('expression') not in ('neutral', 'gentle_smile', 'attentive')
                or row.get('motion') not in ('still', 'subtle_nod')
                or row.get('tone') not in ('calm', 'friendly', 'thoughtful')))): 
        raise ValueError('exact director proposal required')
    reply = row.get('reply')
    if (type(reply) is not str or not reply.strip() or len(reply) > 160
            or any(ord(c) < 32 and c not in '\t\n\r' for c in reply)):
        raise ValueError('bounded director reply required')
    return reply


class DirectorFailure(RuntimeError):
    """Correlated allowlisted metadata only; never attach the child response."""
    def __init__(self, diagnostic):
        self.diagnostic = dict(diagnostic)
        super().__init__('director failed; no fallback: ' + diagnostic['code'])


def _director_failure_diagnostic(row, turn_id):
    if (type(row) is not dict or set(row) != {'type', 'turn_id', 'code', 'diagnostic'}
            or row.get('type') != 'error' or row.get('turn_id') != turn_id
            or row.get('code') not in ('director_failed', 'canceled')):
        raise RuntimeError('director failed; no fallback')
    diagnostic = row['diagnostic']
    codes = {'provider_http_error', 'provider_network_error', 'assistant_error', 'tool_error',
             'missing_proposal', 'canceled', 'aborted', 'runtime_error', 'contract_error'}
    if (type(diagnostic) is not dict or set(diagnostic) != {'schema', 'code', 'http_status'}
            or diagnostic.get('schema') != 'h3-director-failure-v1'
            or type(diagnostic.get('code')) is not str or diagnostic['code'] not in codes
            or (diagnostic['http_status'] is not None and
                (type(diagnostic['http_status']) is not int or not 400 <= diagnostic['http_status'] <= 599))
            or (diagnostic['code'] == 'provider_http_error' and diagnostic['http_status'] is None)
            or (diagnostic['code'] == 'provider_network_error' and diagnostic['http_status'] is not None)
            or (row['code'] == 'canceled' and diagnostic['code'] != 'canceled')):
        raise RuntimeError('director failed; no fallback')
    return dict(diagnostic)


class PiDirector:
    """Exact existing two-custom-tool RPC child, without budget/mailbox writes."""
    def __init__(self, token, *, deadline, max_turns=2, visual_intents=False, long_session=False, stable_selfie=False):
        if type(long_session) is not bool or not 1 <= max_turns <= (1000 if long_session else 2) or not token or deadline <= time.monotonic():
            raise ValueError('finite director configuration required')
        if type(visual_intents) is not bool:raise ValueError('explicit visual director contract required')
        if type(stable_selfie) is not bool or (stable_selfie and not visual_intents):
            raise ValueError('stable selfie requires visual director transport')
        self.visual_intents=visual_intents
        self.stable_selfie=stable_selfie
        node = shutil.which('node')
        if not node:
            raise RuntimeError('existing Node required; no install')
        self.deadline = deadline
        self.child = subprocess.Popen(
            [node, str(Path(__file__).parent / 'director/agent.mjs')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=dict(child_environment(token, max_turns, **({'long_session':True} if long_session else {})), **({'H3_DIRECTOR_VISUAL_INTENTS':'1'} if visual_intents else {}), **({'H3_DIRECTOR_STABLE_SELFIE':'1'} if stable_selfie else {})), start_new_session=True, bufsize=0)
        self.buffer = bytearray()
        try:
            ready = self._read()
            if (ready.get('type') != 'ready' or ready.get('model') != MODEL
                    or ready.get('tools') != ['h3_get_state', 'h3_submit_turn']
                    or ready.get('persistent_session') is not False
                    or (visual_intents and ready.get('visual_intents') is not True)
                    or (stable_selfie and ready.get('stable_selfie') is not True)): 
                raise RuntimeError('exact director unavailable')
        except BaseException:
            self.close()
            raise

    def _read(self):
        import os
        while b'\n' not in self.buffer:
            left = self.deadline - time.monotonic()
            if left <= 0 or not select.select([self.child.stdout], [], [], left)[0]:
                raise TimeoutError('director deadline')
            block = os.read(self.child.stdout.fileno(), 4096)
            if not block:
                raise RuntimeError('director closed')
            self.buffer.extend(block)
            if len(self.buffer) > 32768:
                raise RuntimeError('director response bound')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return json.loads(line)

    def turn(self, text):
        turn_id = uuid.uuid4().hex
        packet = json.dumps(dict(op='turn', turn_id=turn_id, text=text), ensure_ascii=False).encode()+b'\n'
        self.child.stdin.write(packet)
        row = self._read()
        if type(row) is dict and row.get('type') == 'error':
            raise DirectorFailure(_director_failure_diagnostic(row, turn_id))
        if (type(row) is not dict or row.get('type') != 'turn' or row.get('turn_id') != turn_id
                or row.get('model') != MODEL or row.get('effects_executed') is not False):
            raise RuntimeError('director failed; no fallback')
        proposal_reply(row, visual_intents=getattr(self, 'visual_intents', False),
                       stable_selfie=getattr(self, 'stable_selfie', False))
        return row

    def close(self):
        if self.child.poll() is None:
            try:
                self.child.stdin.write(b'{"op":"cancel"}\n')
                self.child.stdin.close()
                self.child.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.child.terminate()
                try:
                    self.child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=2)
        self.child.stdout.close()
        self.buffer.clear()


def _asr_input(item):
    from research.h3_personal.asr_source_contract import validate_actual_asr
    if type(item) is not dict or set(item) != {'text', 'asr'}:
        raise ValueError('exact ASR input required')
    text = item['text']
    if type(text) is not str or not text.strip() or len(text.encode()) > 2000:
        raise ValueError('bounded text required')
    return text, validate_actual_asr(item['asr'])


def visual_turn_payload(payload, *, visual_intent):
    """Version a proposable payload for a visual turn without rewriting provenance.

    A visual turn is still an input turn: ASR turns keep their validated ASR
    receipt/origin and become h3-relay-asr-turn-v2, fresh typed turns become
    h3-relay-typed-turn-v2. Input provenance is never relabelled, so a missing
    or mutated receipt fails closed here as well as at the worker.
    """
    from research.h3_personal.conversation_conditioning import validate_visual_intent
    from research.h3_personal.asr_source_contract import asr_origin, validate_actual_asr
    intent = validate_visual_intent(visual_intent)
    if type(payload) is not dict:
        raise ValueError('exact turn payload required')
    asr_used = payload.get('asr_used')
    if type(asr_used) is not bool:
        raise ValueError('explicit input provenance required')
    if asr_used:
        if payload.get('schema') != 'h3-relay-asr-turn-v1' or 'asr' not in payload:
            raise ValueError('ASR receipt required for an ASR turn')
        asr = validate_actual_asr(payload['asr'])
        if payload.get('origin') != asr_origin(asr):
            raise ValueError('ASR origin/receipt mismatch')
        return dict(payload, schema='h3-relay-asr-turn-v2', asr=asr, visual_intent=intent)
    if payload.get('schema') != 'h3-relay-typed-turn-v1' or 'asr' in payload:
        raise ValueError('typed turn cannot carry an ASR receipt')
    if payload.get('origin') != 'fresh_typed_text':
        raise ValueError('fresh typed origin required')
    return dict(payload, schema='h3-relay-typed-turn-v2', visual_intent=intent)


def build_receiver_pump(pump_factory, transport, binding, root, sink, *, deadline, prefetch_receive):
    """Construct the receive pump; prefetch is opt-in and applies only to the default."""
    if type(prefetch_receive) is not bool:
        raise ValueError('explicit prefetch_receive bool required')
    if prefetch_receive and pump_factory is ReceiverPump:
        return pump_factory(transport, binding, root, sink, deadline=deadline, prefetch=True)
    return pump_factory(transport, binding, root, sink, deadline=deadline)


def wait_native_preroll(sink, progress, *, deadline, reserve_samples, check):
    """Observe source1 + local AV eligibility, never manufacture generation.

    A streaming snapshot races EOS. Only server cancelled terminal establishes
    native cancellation. Terminal-before-trigger is a normal unexercised case.
    No speech PCM, artificial native delay or inference retry is used here.
    """
    import math
    if (not callable(progress) or not callable(check) or type(reserve_samples) is not int
            or not 1<=reserve_samples<=96000 or type(deadline) not in (int,float)
            or not math.isfinite(deadline)): 
        raise ValueError('bound native progress and preroll policy required')
    until=min(deadline,time.monotonic()+120)
    last_ns=0;last_epoch=0;last_samples=0;polls=0;terminal=None
    while True:
        check()
        if time.monotonic()>=until:raise TimeoutError('native preroll/local AV eligibility deadline')
        row=terminal if terminal is not None else progress(1,min(until,time.monotonic()+2))
        polls+=int(terminal is None)
        if (type(row) is not dict or row.get('schema')!='h3-native-progress-v1'
                or type(row.get('epoch')) is not int or row['epoch'] not in (0,1)
                or row['epoch']<last_epoch or type(row.get('observed_monotonic_ns')) is not int
                or row['observed_monotonic_ns']<last_ns or type(row.get('samples')) is not int
                or not 0<=row['samples']<=48000*30 or row['samples']<last_samples
                or type(row.get('terminal')) is not bool
                or row.get('state') not in ('idle','streaming','cancel_requested','cancelled','iterator_exhausted')):
            raise ValueError('stale or invalid source1 native progress')
        last_ns=row['observed_monotonic_ns'];last_epoch=row['epoch'];last_samples=row['samples']
        first=sink.first_av_submission_monotonic
        eligible=first is not None
        if eligible and (type(first) not in (int,float) or not math.isfinite(first)
                         or not 0<=first<=time.monotonic()):
            raise ValueError('actual local first AV eligibility required')
        if row['terminal'] or row['state'] in ('cancelled','iterator_exhausted','cancel_requested'):
            terminal=dict(row)
            if eligible:
                return dict(schema='h3-native-preroll-trigger-v1',source_epoch=1,
                    outcome='already_terminal_or_cancel_pending',native_progress=dict(row),
                    local_first_av_monotonic=first,observed_local_monotonic=time.monotonic(),polls=polls)
        elif eligible and row['epoch']==1 and row['state']=='streaming' and row['samples']>=reserve_samples:
            return dict(schema='h3-native-preroll-trigger-v1',source_epoch=1,outcome='eligible_snapshot',
                native_progress=dict(row),local_first_av_monotonic=first,
                observed_local_monotonic=time.monotonic(),polls=polls)
        time.sleep(min(.05,max(0,until-time.monotonic())))  # metadata polling, never native model delay


def run_conversation(transport, binding, root, turns, director, *, ready,
                     max_seconds=120, max_turns=2, allow_obs=False, obs_port=None,
                     cancel=None, sink_factory=None, pump_factory=ReceiverPump,
                     prefetch_receive=False,
                     input_source=None, direct_audio_binary=None, audio_delay_seconds=0.2,
                     hold_listening=False, start_on_first_av=False,
                     ten_minute_test=False, headless_interrupt=False, visual_intents=False, review_av=False,
                     attended_visual=False, today_pause_on_underflow=False, long_session=False, stable_selfie=False,
                     native_cancel_trigger=None, native_progress=None, native_preroll_samples=None):
    """Run <=2 pre-supplied fresh typed strings (never an unbounded input reader).

    With input_source, turns must be None. The caller-owned cooperative callable
    is invoked with keyword context={session, epoch} (next turn epoch) and the
    absolute monotonic deadline, after receiver startup, at most max_turns times.
    It returns None to finish or exactly {text, asr}; it must enforce the deadline
    and observe the caller's cancellation event itself. No thread or implicit
    close is used for source calls; the caller owns source cleanup. Late/canceled
    completions are rejected before director/relay use. Only provenance, never
    the transcript, reaches the relay.

    Explicit hold_listening keeps the live receiver listening after input ends,
    until deadline minus two seconds, reserving the final two seconds for drain.
    This opt-in mode is bounded to five minutes and never requests more input.
    ten_minute_test explicitly permits exactly 600 seconds only with both hold
    and first-AV clock enabled; the parent launcher must opt in for that mode.
    start_on_first_av reserves that full duration within the READY lease and
    waits at most 120 seconds for actual sink AV submission before input capture.

    READY acquisition and sourceSpeechInfo
    control delivery are NOT implemented here: a production orchestrator must
    acquire live READY and route worker source metadata to the receiver.
    READY must explicitly bind session/profile/clock epoch and expiry. Transport
    must already be authenticated/open; no connect/claim is performed here.
    director.turn/close must obey the same finite deadline (PiDirector does).
    On failure/cancel send cancel_pending then stop when possible, never EOS.
    Only a genuine producer final pair completes the receiver.
    """
    early=native_cancel_trigger=='native-preroll-v1'
    if (native_cancel_trigger not in (None,'native-preroll-v1') or
            (early and (not headless_interrupt or ready.get('native_cancel_trigger')!=native_cancel_trigger
                        or ready.get('native_tts_cancel') is not True or not callable(native_progress)))
            or (not early and (native_progress is not None or native_preroll_samples is not None
                               or 'native_cancel_trigger' in ready))):
        raise ValueError('explicit native-preroll READY/progress binding required')
    stable_devicefree = stable_selfie is True and attended_visual is False
    if type(review_av) is not bool or (review_av and
            (not visual_intents or (not stable_devicefree and ready.get('paired_cuts') is not True))):
        raise ValueError('review AV needs explicit visual/cut READY')
    if type(visual_intents) is not bool or type(headless_interrupt) is not bool:
        raise ValueError('explicit visual/headless selection required')
    # Attended is the already-verified device route: real Comet ASR input and the
    # split output (video to OBS, original PCM to the audio device). Turn
    # interruption, native preroll, paired review cuts, periodic anchors and the
    # caller browser sink are absent, and this runner never fabricates a synthetic
    # onset. Capture timing stays operator/caller-owned; no automated gate.
    if type(attended_visual) is not bool:
        raise ValueError('explicit attended visual capability required')
    if visual_intents and attended_visual is headless_interrupt and not stable_devicefree:
        raise ValueError('visual intents require exactly one explicit headless or attended route')
    if visual_intents and ready.get('visual_intents') is not True:
        raise ValueError('visual intents require explicitly bound visual READY')
    if (type(stable_selfie) is not bool or (stable_selfie and not visual_intents)
            or ready.get('stable_selfie', False) is not stable_selfie):
        raise ValueError('stable selfie requires explicitly bound visual READY')
    if stable_selfie and (headless_interrupt
            or ready.get('anchor_policy') != 'conversation-tail-every3-v1'
            or ready.get('turn_interruptions', False) is not False
            or 'paired_cuts' in ready):
        raise ValueError('stable selfie requires exact conversation anchor without interruption/cuts')
    if stable_devicefree and (input_source is not None or allow_obs or obs_port is not None
            or direct_audio_binary is not None or sink_factory is not None or hold_listening
            or type(turns) not in (list, tuple) or len(turns) != 2
            or ready.get('attended_visual', False) is not False):
        raise ValueError('device-free stable selfie requires two typed turns and no output adapters')
    if attended_visual:
        if (ready.get('attended_visual') is not True
                or (stable_selfie and ready.get('anchor_policy') != 'conversation-tail-every3-v1')
                or (not stable_selfie and 'anchor_policy' in ready)):
            raise ValueError('attended visual requires matching explicit ancestry READY')
        if (review_av or native_cancel_trigger is not None or native_progress is not None
                or native_preroll_samples is not None or input_source is None
                or allow_obs is not True or obs_port is None or direct_audio_binary is None):
            raise ValueError('attended visual requires the real input source plus the split '
                             'OBS video and PCM audio output, and no headless/preroll/review modes')
    if headless_interrupt and (attended_visual or input_source is not None or allow_obs or obs_port is not None
            or direct_audio_binary is not None or sink_factory is not None
            or type(turns) not in (list, tuple) or len(turns) != 2 or hold_listening):
        raise ValueError('device-free headless interruption requires two typed turns and no output adapters')
    if (type(hold_listening) is not bool or type(start_on_first_av) is not bool
            or type(ten_minute_test) is not bool or type(prefetch_receive) is not bool):
        raise ValueError('listening clock options must be bool')
    if ten_minute_test and (not hold_listening or not start_on_first_av
                            or type(max_seconds) not in (int, float)
                            or max_seconds != 600):
        raise ValueError('ten minute test requires hold, first AV clock and exactly 600 seconds')
    if start_on_first_av and not hold_listening:
        raise ValueError('first AV clock requires hold_listening')
    if type(long_session) is not bool or (long_session and
            (not attended_visual or not today_pause_on_underflow or ready.get('long_session') is not True)):
        raise ValueError('three-hour relay requires bound attended pause selection')
    if (type(max_turns) is not int or not 1 <= max_turns <= (1000 if long_session else 2)
            or type(max_seconds) not in (int, float)
            or not 0 < max_seconds <= (10800 if long_session else 600 if ten_minute_test or not hold_listening else 300)
            or (input_source is None and
                (type(turns) not in (list, tuple) or not 1 <= len(turns) <= max_turns))
            or (input_source is not None and
                (not callable(input_source) or turns is not None))):
        raise ValueError('finite RAM typed turns required')
    if input_source is None and any(type(t) is not str or not t.strip() or len(t.encode()) > 2000 for t in turns):
        raise ValueError('bounded typed text required')
    if (ready.get('status') != 'ready' or ready.get('session') != binding.session
            or ready.get('profile_sha256') != binding.profile_sha256
            or ready.get('epoch') != binding.epoch
            or not time.time() < ready.get('expires_at', 0) <= time.time()+(10800 if long_session else 3600)):
        raise ValueError('explicit live READY/profile metadata required')
    if (transport._session != binding.session or transport._role != 'consumer'
            or not transport._opened or transport._closed or not transport._attested):
        raise ValueError('opened authenticated consumer required')
    if allow_obs is not True and obs_port is not None:
        raise PermissionError('OBS requires explicit allow_obs approval')
    if type(today_pause_on_underflow) is not bool or (today_pause_on_underflow and
            (not attended_visual or direct_audio_binary is None)):
        raise ValueError('today pause workaround requires attended split output')
    audio_pin = None
    if direct_audio_binary is not None:
        if allow_obs is not True or obs_port is None:
            raise PermissionError('split route requires explicit OBS approval and port')
        audio_pin = direct_audio_pin(direct_audio_binary, audio_delay_seconds)
    elif audio_delay_seconds != 0.2:
        raise ValueError('audio delay requires direct audio binary')
    lease_deadline = time.monotonic()+ready['expires_at']-time.time()
    deadline = (lease_deadline if start_on_first_av else
                min(time.monotonic()+max_seconds, lease_deadline))
    if isinstance(director, PiDirector):
        director.deadline = min(director.deadline, deadline)
    commands = Commands(transport, binding.session,
                        **({'turn_interruptions':True} if headless_interrupt else {}),
                        **({'long_session':True} if long_session else {}))
    controller = None
    epoch = 0
    review_archive = None
    review_direction = None
    review_intent_epoch = 0
    trigger_evidence = None
    result = []
    errors = []
    sink = None
    thread = None
    stopped = False
    from research.h3_personal.phase_timing import PhaseTiming
    from research.h3_personal.direct_asr_input import DirectASRInput
    timing = PhaseTiming()
    previous_timing = None
    if isinstance(input_source, DirectASRInput):
        previous_timing = input_source.phase_timing
        input_source.phase_timing = timing
    try:
        if audio_pin is not None:
            from research.h3_personal.split_av_sink import SplitAVSink
            factory = sink_factory or SplitAVSink
        else:
            if headless_interrupt:
                from research.h3_personal.epoch_av_sink import EpochAVSink
                factory = EpochAVSink
            else:
                factory = sink_factory or (OBSAVSink if allow_obs else HeadlessAVSink)
        kwargs = dict(max_frames=124, max_total_frames=binding.max_total_frames,
                      timeout=min(30, max_seconds))
        if headless_interrupt:kwargs.update(paced=True, observe_output=True)
        if review_av:
            from research.h3_personal.headless_av_sink import ReviewArchive
            review_archive = ReviewArchive(Path(root)/'review', deadline=deadline)
            kwargs.update(review_archive=review_archive)
            if not stable_devicefree:kwargs['paired_cuts'] = True
        if allow_obs:
            kwargs.update(port=obs_port, playback_approved=True)
        if audio_pin is not None:
            if direct_audio_pin(direct_audio_binary, audio_delay_seconds) != audio_pin:
                raise ValueError('direct audio binary changed')
            kwargs.update(audio_binary=audio_pin['audio_binary'],
                          audio_delay_seconds=audio_pin['audio_delay_seconds'])
        if long_session:
            kwargs['long_session'] = True
        if today_pause_on_underflow:
            kwargs['today_pause_on_underflow'] = True
        sink = factory(root, binding, **kwargs)
        if headless_interrupt:
            from research.h3_personal.turn_control import TurnController
            controller = TurnController(commands,sink,deadline=deadline,max_turns=2)
        pump = build_receiver_pump(pump_factory, transport, binding, root, sink,
                                   deadline=deadline, prefetch_receive=prefetch_receive)

        def receive():
            try:
                outcome = pump.run()
                result.append(outcome)
                try:
                    # Bounded local-clock receipt only; never a remote claim.
                    evidence = getattr(pump, 'timing', None)
                    if callable(evidence):
                        persist(root, 'pump-timing.json', evidence())
                    events = getattr(sink, 'timing', None)
                    if isinstance(events, dict) and isinstance(events.get('events'), list):
                        persist(root, 'sink-events.json', dict(clock='local_monotonic',
                                                               events=events['events'][:144]))
                except BaseException:
                    pass  # Evidence must never turn a completed session into a failure.
            except BaseException as exc:
                from research.h3_personal.local_av_diagnostics import failure_evidence
                diagnostic = _receiver_diagnostic(exc)
                failure = failure_evidence(getattr(transport, 'last_failure', None))
                evidence = {'stack': diagnostic}
                if failure:
                    evidence['transport_failure'] = failure
                errors.append(diagnostic)
                try:
                    persist(root, 'receiver-failure.json', evidence)
                except OSError:
                    pass  # Never hide the original receiver failure.
                if failure:
                    try:
                        persist(root, 'receiver-transport-failure.json', failure)
                    except OSError:
                        pass  # Original receiver failure remains authoritative.

        thread = threading.Thread(target=receive, name='h3-mac-receiver')
        thread.start()
        if start_on_first_av:
            import math
            first_wait_deadline = min(time.monotonic()+120,
                                      lease_deadline-max_seconds)
            while True:
                if errors:
                    raise RuntimeError(f'receiver failed; no replay: {errors[0]}')
                if result or not thread.is_alive() or (cancel and cancel.is_set()):
                    raise RuntimeError('conversation canceled or receiver ended')
                now = time.monotonic()
                if now >= first_wait_deadline:
                    raise TimeoutError('first AV submission deadline')
                first = sink.first_av_submission_monotonic
                if first is not None:
                    if (type(first) not in (int, float) or
                            not math.isfinite(first) or not 0 <= first <= now):
                        raise ValueError('finite actual first AV timestamp required')
                    deadline = min(first+max_seconds, lease_deadline)
                    director.deadline = min(director.deadline, deadline)
                    if hasattr(pump, 'deadline'):
                        pump.deadline = deadline
                    receiver = getattr(pump, 'receiver', None)
                    if hasattr(receiver, 'deadline'):
                        receiver.deadline = deadline
                    print(json.dumps(dict(type='conversation_playback_clock',
                                          started=first, clock='monotonic',
                                          max_seconds=max_seconds)), flush=True)
                    break
                thread.join(min(.05, first_wait_deadline-now))
        total_turns = max_turns if input_source is not None else len(turns)
        for index in range(total_turns):
            if errors:
                raise RuntimeError(f'receiver failed; no replay: {errors[0]}')
            if result or time.monotonic() >= deadline or (cancel and cancel.is_set()):
                raise RuntimeError('conversation canceled or receiver ended')
            if controller is not None:
                if index == 1:
                    if early:
                        def check_progress():
                            if errors or result or not thread.is_alive() or (cancel and cancel.is_set()):
                                raise RuntimeError('receiver ended during native progress observation')
                        trigger_evidence=wait_native_preroll(sink,native_progress,deadline=deadline,
                            reserve_samples=native_preroll_samples,check=check_progress)
                        persist(root,'native-trigger.json',trigger_evidence)
                    else:
                        sink.wait_for_speech_output(1,deadline)
                        # Legacy actual decoded old-voice trial remains unchanged.
                        time.sleep(.2)
                token = controller.begin_input()
            provenance = dict(schema='h3-relay-typed-turn-v1',
                              origin='fresh_typed_text', asr_used=False)
            if input_source is not None:
                with timing.stage('input'):
                    item = input_source(context=dict(session=binding.session,
                                                     epoch=epoch+1), deadline=deadline)
                if errors:
                    raise RuntimeError(f'receiver failed; no replay: {errors[0]}')
                if result or time.monotonic() >= deadline or (cancel and cancel.is_set()):
                    raise RuntimeError('stale input completion')
                if item is None:
                    break
                from research.h3_personal.asr_source_contract import asr_origin
                text, asr = _asr_input(item)
                provenance = dict(schema='h3-relay-asr-turn-v1',
                                  origin=asr_origin(asr),
                                  asr_used=True, asr=asr)
            else:
                with timing.stage('input'):
                    text = turns[index]
            if controller is not None and not controller.commit_input(token):
                raise RuntimeError('stale finalized input')
            with timing.stage('pi'):
                proposal = director.turn(text)
            if errors:
                raise RuntimeError(f'receiver failed; no replay: {errors[0]}')
            if time.monotonic() >= deadline or (cancel and cancel.is_set()) or result:
                raise RuntimeError('stale director completion')
            reply = proposal_reply(proposal, visual_intents=visual_intents, stable_selfie=stable_selfie)
            epoch += 1
            with timing.stage('publication'):
                payload = dict(**provenance, director_model=MODEL, reasoning_effort='none',
                               reply=reply, previous_audio_reused=False)
                if visual_intents:
                    payload = visual_turn_payload(payload, visual_intent=proposal['visual_intent'])
                if controller is None:commands.send('turn', epoch, payload)
                elif not controller.publish_reply(token,payload):
                    raise RuntimeError('stale Pi publication')
                if review_archive is not None:
                    from research.h3_personal.conversation_conditioning import resolve_visual_intent
                    proposed = payload['visual_intent']
                    review_direction = resolve_visual_intent(review_direction, proposed)
                    if proposed['instruction'] is not None:
                        review_intent_epoch = epoch
                    review_archive.capture_visual_direction(proposed=proposed, active=review_direction,
                        intent_epoch=review_intent_epoch, target_epoch=epoch)
                note = getattr(input_source, 'note_reply_published', None)
                if callable(note):
                    note()
        if hold_listening:
            while True:
                if errors:
                    raise RuntimeError(f'receiver failed; no replay: {errors[0]}')
                if result or not thread.is_alive() or (cancel and cancel.is_set()):
                    raise RuntimeError('conversation canceled or receiver ended')
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('finite conversation stopped')
                if now >= deadline - 2:
                    break
                thread.join(min(.05, deadline - 2 - now))
        commands.send('stop', epoch, {})
        stopped = True
        while thread.is_alive():
            if time.monotonic() >= deadline or (cancel and cancel.is_set()):
                raise TimeoutError('finite conversation stopped')
            thread.join(min(.05, max(0, deadline-time.monotonic())))
        if errors or not result:
            detail = errors[0] if errors else 'no result'
            raise RuntimeError(f'receiver failed; no replay: {detail}')
        timing.completed = True
        if controller is not None:
            report = sink.timing
            cut = report['discontinuities'][0]
            new = report['decoded_source_output'].get('2')
            if not cut['actual_decoded_sink_observed'] or new is None:
                raise ValueError('actual sink cessation and new-source output required')
            if early:
                result[0]['native_trigger']=trigger_evidence
                result[0]['interruption_kind']='generation_preparation_cancel_not_heard_speech_barge_in'
                result[0]['presentation_interrupted']=True
            result[0].update(sink_timing=report,
                stop_criteria=dict(measurement='old_decoder_stdout_EOF_not_selector',
                    upper_bound_seconds=cut['stop_upper_bound_seconds'],
                    last_old_pcm_monotonic=cut['old_sink_last_pcm_monotonic'],
                    passed=cut['stop_upper_bound_seconds'] <= .2),
                response_criteria=dict(measurement='new_source_decoded_PCM_stdout',
                    seconds=new['first_monotonic']-cut['detected_monotonic'],
                    functional_reentry_passed=new['first_monotonic'] >= cut['old_sink_eof_monotonic'],
                    latency_acceptance='measured_only_no_product_latency_threshold'),
                fresh_speech=False, devices_used=False)
        return result[0]
    finally:
        try:
            if not stopped:
                try:
                    commands.send('cancel_pending', epoch, {})
                    commands.send('stop', epoch, {})
                except Exception:
                    pass
            director.close()
            if thread is not None and thread.is_alive():
                transport.abort()
                if sink is not None:
                    sink.cancel()
                thread.join(3)
            if sink is not None:
                sink.cancel()
                try:
                    _persist_audio_evidence(root,sink)
                except Exception:
                    print(json.dumps({'type':'encoder_audio_evidence','status':'unavailable'}),flush=True)
                try:
                    _persist_output_timing(root, sink)
                except Exception:
                    print(json.dumps({'type':'split_av_timing','status':'unavailable'}),flush=True)
            commands.close()
            transport.close()
            if thread is not None and thread.is_alive():
                raise RuntimeError('receiver did not terminate')
        except BaseException:
            timing.completed = False
            raise
        finally:
            if isinstance(input_source, DirectASRInput):
                input_source.phase_timing = previous_timing
            import sys
            primary = sys.exc_info()[1]
            try:
                timing.persist(root)
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note('stage timing unavailable: '+type(error).__name__)
