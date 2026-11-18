"""Finite typed-text/Pi-reply seam. No startup, mailbox, upload or batch render.

Caller supplies an opened authenticated producer SocketAdapter and ready TTS
(hf_h3_live_voice.TTS signature). Default conversion is bounded FFmpeg to
48k mono S16LE, never the native runtime's S32LE helper.
Turn payload accepts typed receipts or explicitly pinned native-completed ASR
receipts; no raw user text or transcript.
Pi execution belongs to the authenticated input owner; this validates its receipt,
not independent proof of remote inference. Default complete watermarked synthesis
precedes incremental H3. Explicit sentence_incremental publishes each separately
synthesized sentence into live ingress; this is NOT native sample-streaming TTS
or streaming ASR.
"""
from dataclasses import asdict, replace
import hashlib
import math
import json
import re
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

from .conversation_conditioning import role_profile_digest
from .director_irodori import SOURCE, MODEL, CODEC
from .live_conversation import LiveConversation
from .live_native_backend import LiveNativeBackend
from .live_relay_commands import Commands
from .paired_av_pump import ProducerPump
from .streaming_pcm import PCMIngress, PCMStreamBinding, PCMChunk

PI_MODEL = 'Qwen/Qwen3.8-27B:cerebras'
NATIVE_CANCEL_TERMINAL_SECONDS = 15


def _emit_producer_evidence(producer, adapter, *, failed):
    """Best-effort metadata only, captured before worker/native cleanup.

    Never stringify exceptions or forward arbitrary snapshot/timing dictionaries.
    Older pump/adapter mocks need not implement any diagnostic interfaces.
    """
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    try:
        receipt = producer.timing()
        if type(receipt) is dict:
            windows = []
            keys = {'index', 'credit_wait_started', 'pull_started', 'packet_ready',
                    'video_bytes', 'pcm_bytes', 'send_started', 'send_enqueued',
                    'ack_verified', 'ack_sink_accepted', 'ack_discarded_interrupted',
                    'ack_partially_presented_interrupted'}
            rows=receipt.get('windows', ())
            if type(rows) not in (list,tuple):rows=()
            for row in rows[:144]:
                if type(row) is dict:
                    windows.append({k: v for k, v in row.items() if k in keys and number(v)})
            print(json.dumps(dict(type='h3_producer_timing', clock='local_monotonic',
                origin_monotonic=receipt.get('origin_monotonic') if number(receipt.get('origin_monotonic')) else None,
                send_semantics='local_enqueue_not_peer_delivery', windows=windows),
                allow_nan=False), flush=True)
    except BaseException:
        pass
    if not failed:
        return
    evidence = dict(type='h3_producer_failure', failed=True)
    try:
        snapshot = producer.sender.snapshot()
        if type(snapshot) is dict:
            evidence['sender'] = {k: v for k, v in snapshot.items()
                if k in {'reserved', 'verified', 'acked', 'discarded', 'pending', 'inflight_bytes',
                         'frame_stop', 'partial_acked'} and number(v)}
            if type(snapshot.get('final')) is bool:
                evidence['sender']['final'] = snapshot['final']
            evidence['sender']['terminal'] = snapshot.get('terminal') is not None
    except BaseException:
        pass
    try:
        if number(producer.highwater):
            evidence['queue_highwater'] = producer.highwater
        evidence['queue_depth'] = len(producer._send_queue)
        evidence['transport_failed'] = producer._error is not None
    except BaseException:
        pass
    try:
        failure = adapter.last_failure
        if type(failure) is dict:
            evidence['transport_failure_recorded'] = True
            status = failure.get('http_status')
            if type(status) is int and 100 <= status <= 599:
                evidence['http_status'] = status
    except BaseException:
        pass
    try:
        print(json.dumps(evidence, allow_nan=False), flush=True)
    except BaseException:
        pass


def validate_turn(row, expected_asr=None, *, visual_intents=False, stable_selfie=False):
    p = row['payload']
    if type(visual_intents) is not bool:raise ValueError('explicit visual turn contract required')
    if type(stable_selfie) is not bool or (stable_selfie and not visual_intents):
        raise ValueError('stable selfie requires explicit visual turn contract')
    version='v2' if visual_intents else 'v1'
    is_asr = p.get('schema') == 'h3-relay-asr-turn-'+version
    fields = {'schema','origin','asr_used','director_model','reasoning_effort',
              'reply','previous_audio_reused'}
    if visual_intents:
        from .conversation_conditioning import validate_visual_intent
        validate_visual_intent(p.get('visual_intent'))
        if stable_selfie and p['visual_intent']['instruction'] is not None:
            raise ValueError('stable selfie forbids movement instruction')
        fields.add('visual_intent')
    if is_asr:
        fields.add('asr')
        from research.h3_personal.asr_source_contract import validate_actual_asr, asr_origin
        if expected_asr is None:
            raise ValueError('immutable expected ASR required')
        validate_actual_asr(p.get('asr'), expected_asr)
    if (set(p) != fields
            or p['schema'] != ('h3-relay-asr-turn-'+version if is_asr else 'h3-relay-typed-turn-'+version)
            or p['origin'] != (asr_origin(p['asr']) if is_asr else 'fresh_typed_text')
            or p['director_model'] != PI_MODEL
            or p['asr_used'] is not is_asr or p['previous_audio_reused'] is not False
            or p['reasoning_effort'] != 'none'
            or type(p['reply']) is not str or not p['reply'].strip() or len(p['reply']) > 160):
        raise ValueError('bounded fresh input/Pi receipt required')
    return p['reply']


def _visual_window_evidence(receipt, *, stable_selfie=False, anchor_sha256=None):
    """Export existing native scalar evidence, never infer a missing selection."""
    if type(stable_selfie) is not bool:
        raise ValueError('explicit stable selfie evidence selection required')
    value = dict(sequence=receipt['sequence'], **receipt['visual_conditioning'])
    if stable_selfie:
        from .visual_ancestry_policy import validate_selection
        value['visual_ancestry_selection'] = validate_selection(
            receipt.get('visual_ancestry_selection'), receipt['sequence'], anchor_sha256)
    return value


def canonical_ffmpeg_pcm(wav, deadline):
    """Bounded conversion, no truncation, alternate codec or sample-rate fallback."""
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('PCM conversion deadline')
    with tempfile.TemporaryDirectory(dir=Path(wav).parent) as tmp:
        output = Path(tmp)/'canonical.pcm'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-n',
            '-i',str(wav),'-map','0:a:0','-ac','1','-ar','48000',
            '-c:a','pcm_s16le','-f','s16le','-fs','2880002',str(output)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=True, timeout=remaining)
        if time.monotonic() >= deadline:
            raise TimeoutError('PCM conversion deadline')
        if not 0 < output.stat().st_size <= 2880000:
            raise ValueError('canonical PCM exceeds finite duration; never truncate')
        return output.read_bytes()


def validate_complete_wav(result, reply, turn_id, directory):
    """Validate a fresh sentence/reply receipt before any PCM publication."""
    r = result['receipt']; wav = Path(result['wav']).resolve()
    if (result['turn_id'] != turn_id or wav != (directory/'reply.wav').resolve()
            or not wav.is_file() or wav.stat().st_size > 16*1024*1024
            or r.get('watermark_applied') is not True
            or (r.get('source_revision'),r.get('model_revision'),r.get('codec_revision')) != (SOURCE,MODEL,CODEC)
            or r.get('text_sha256') != hashlib.sha256(reply.encode()).hexdigest()
            or r.get('sha256') != hashlib.sha256(wav.read_bytes()).hexdigest()):
        raise ValueError('fresh complete watermarked WAV receipt required')
    return wav


def complete_pcm(result, reply, turn_id, directory, converter, deadline):
    wav = validate_complete_wav(result, reply, turn_id, directory)
    pcm = converter(wav, deadline)
    if type(pcm) is not bytes or not 0 < len(pcm) <= 2880000 or len(pcm)%2:
        raise ValueError('canonical original 48k mono S16LE required; never truncate')
    return pcm


class TurnInterruptionGate:
    """Constant-bounded worker turn quarantine, not sink invalidation.

    The input owner must invalidate local A/V before sending interrupt_turn.
    Never use current() as the native TTS drain predicate: an already-started
    native request must consume verified END so its healthy Gate can be reused.
    """
    def __init__(self):
        self._lock=threading.Lock()
        self._accepted={};self._invalidated=0;self._presentation=0
        self._interrupts=0;self._quarantined=0;self._skipped=0

    def accept(self, epoch):
        with self._lock:
            if (type(epoch) is not int or not 1<=epoch<=16 or epoch<=max(self._accepted,default=0)
                    or len(self._accepted)>=2):
                raise ValueError('at most two monotonically increasing worker turns')
            self._accepted[epoch]=False

    def invalidate(self, row):
        p=row.get('payload')
        with self._lock:
            if (row.get('kind')!='interrupt_turn' or type(row.get('epoch')) is not int
                    or row['epoch']!=max(self._accepted,default=0)
                    or row['epoch']<=self._invalidated or type(p) is not dict
                    or set(p)!={'schema','presentation_epoch'}
                    or p['schema']!='h3-turn-interruption-v1'
                    or type(p['presentation_epoch']) is not int
                    or not self._presentation<p['presentation_epoch']<=2**63-1):
                raise ValueError('current turn and monotonic presentation epoch required')
            self._invalidated=row['epoch'];self._presentation=p['presentation_epoch']
            self._interrupts+=1

    def current(self, epoch):
        with self._lock:
            return type(epoch) is int and epoch in self._accepted and epoch>self._invalidated

    def complete(self, epoch, *, skipped=False):
        if type(skipped) is not bool:raise ValueError('explicit skipped-request boolean required')
        with self._lock:
            if epoch not in self._accepted or self._accepted[epoch]:
                raise ValueError('fresh synthesis completion required')
            self._accepted[epoch]=True
            if skipped:self._skipped+=1
            elif epoch<=self._invalidated:self._quarantined+=1

    def snapshot(self):
        with self._lock:
            return dict(presentation_epoch=self._presentation,
                        invalidated_turn_epoch=self._invalidated,
                        accepted_turns=len(self._accepted),
                        synthesis_drained=all(self._accepted.values()),
                        interrupts=self._interrupts,quarantined_tts_results=self._quarantined,
                        skipped_tts_requests=self._skipped)


def _presentation_packet(packet, epoch):
    """Tag an immutable packet without altering its fixed clock binding."""
    return replace(packet,envelope=replace(packet.envelope,presentation_epoch=epoch))


def _quarantine_suffix(packet, previous, old_epoch, view, *, source_epoch, frontier, current):
    """Derive the omitted prefix from the preceding verified native frontier.

    This does NOT skip a native forward. It admits only a newly generated
    listening suffix; mixed/new speech or an unproved decoder boundary fails.
    """
    from .paired_av import PresentationCut, _bodies
    e=packet.envelope
    if (type(previous) is not dict or type(previous.get('sequence')) is not int
            or e.index!=previous['sequence']+1
            or type(previous.get('committed_frame')) is not int
            or type(previous.get('emitted_frame')) is not int
            or e.frame_start!=previous['emitted_frame']
            or not e.frame_start<previous['committed_frame']<e.frame_stop
            or type(source_epoch) is not int or not 1<=source_epoch<=view['invalidated_turn_epoch']
            or view['presentation_epoch']<=old_epoch
            or type(frontier) is not dict or frontier.get('reserved') is not False
            or frontier.get('windows')!=e.index
            or frontier.get('committed_frame')!=previous['committed_frame']
            or frontier.get('emitted_frame')!=previous['emitted_frame']
            or type(current) is not dict or current.get('sequence')!=e.index
            or current.get('role')!='listening' or current.get('emitted_frame')!=e.frame_stop
            or current.get('committed_before')!=previous['committed_frame']
            or type(current.get('receipt_sha256')) is not str or len(current['receipt_sha256'])!=64):
        raise ValueError('unproved native decoder cut frontier')
    boundary=previous['committed_frame']
    if any(r.kind!='silence' and r.sample_stop>boundary*2000 for r in e.regions):
        raise ValueError('quarantine suffix is not newly committed listening')
    _bodies(e,packet.video,packet.pcm)
    cut=PresentationCut(boundary,old_epoch,source_epoch)
    return replace(packet,envelope=replace(e,presentation_epoch=view['presentation_epoch'],presentation_cut=cut))


def _should_stop_listening(turns, accepted, admission_closed, hold_listening, max_turns=2):
    return ((turns == max_turns and not hold_listening)
            or (admission_closed and turns == accepted))


def _conversation(root, *, binding, context, deadline, ten_minute_test=False, long_session=False):
    """Bound admission to 300s (explicit endurance: 720s) within worker lease.

    Expired leases are rejected by admission, never extended to a positive floor.
    Reserve one window for final listening rather than spending it on idle.
    """
    if type(ten_minute_test) is not bool:
        raise ValueError('explicit boolean ten_minute_test required')
    return LiveConversation(root, binding=binding, context=context,
        max_turns=1000 if long_session else 2, max_idle_windows=binding.max_segments-1,
        timeout=min(10800 if long_session else 720 if ten_minute_test else 300, deadline-time.monotonic()),
        max_total_source_bytes=10800*96000 if long_session else 32*1024*1024,
        extended_endurance=ten_minute_test, long_session=long_session)


def run_worker(runtime, tts, adapter, *, binding, context, root, deadline,
               canonical_pcm=None, on_ready=None, expected_asr=None, visual_ancestry_policy=None,
               ten_minute_test=False, hold_listening=False, tts_mode='complete',
               overlap_send=False, tts_preroll=None,
               media_transport='http-paired-v1', media_adapter=None,
               turn_interruptions=False, visual_intents=False, attended_visual=False, stable_selfie=False, paired_cuts=False,
               native_tts_cancel=False, long_session=False):
    """Own adapter/command reader/conversation; borrow warm runtime and TTS.

    Optional on_ready() runs after backend/conversation initialization, before
    reader threads or the pump start. Callback failures use normal cleanup.
    Hard deadline must also be honored by supplied TTS/converter and adapter.
    At most two fresh epochs, then final listening + certain final sink ACK.
    overlap_send is an explicit bounded generation/enqueue overlap experiment.
    tts_mode='complete' preserves the full-WAV route. Explicit
    'sentence_incremental' validates each marked sentence, preserves native
    integer PCM, and continuously canonicalizes non48k across sentence boundaries.
    'native_voxcpm' requires immutable tts_preroll and a ready stream transport;
    float32 PCM is explicitly converted to S16LE without watermark. Transport
    protocol END is not independent native model EOS. No fallback is allowed.
    No launcher/provider activation is performed here.
    Stop closes admission and drains all previously accepted turns before final
    listening. Cancel independently invalidates queued and in-flight synthesis.
    Ingestion is independent of the sole serial TTS owner. Native
    generation/pump calls remain synchronous: control takes effect at the next
    packet boundary, not by interrupting native GPU work. Cleanup requires the
    supplied TTS to honor its deadline or unblock synthesize when close is called.

    Explicit HTTPS-only turn_interruptions enables interrupt_turn controls.
    They invalidate queued admission without terminating the session or closing
    an active native TTS stream. That request drains END and is quarantined.
    Old native work completes once; one OLD-tagged listening window drains its
    decoder lookahead before fresh speech. The input owner must already have
    invalidated local A/V. Fixed binding.epoch is never a presentation epoch.

    attended_visual is the explicit alternative route: versioned visual turns
    with latest-tail ancestry (or explicitly anchor-bound stable_selfie) and
    NO interrupt_turn controls, so no synthetic interruption schedule exists.
    stable_selfie also permits a device-free typed route without attended mode;
    it rejects every non-null movement proposal before application.
    It is mutually exclusive with
    turn_interruptions and never enables paired/review cuts or native cancel.
    """
    from .voxcpm_endpoint_protocol import CANCEL_PROTOCOL
    if (type(native_tts_cancel) is not bool or (native_tts_cancel and
            (not turn_interruptions or tts_mode != 'native_voxcpm'
             or getattr(tts, 'cancel_protocol', None) != CANCEL_PROTOCOL
             or not callable(getattr(tts, 'request_cancel', None))))):
        raise ValueError('native cancel requires bound v2 client and interruptible native mode')
    if type(long_session) is not bool or (long_session and
            (not attended_visual or turn_interruptions or ten_minute_test
             or media_transport != 'http-paired-v1' or tts_mode != 'native_voxcpm')):
        raise ValueError('long session requires attended native HTTPS only')
    max_turns = 1000 if long_session else 2
    if type(stable_selfie) is not bool or (stable_selfie and
            (visual_intents is not True or turn_interruptions is not False or paired_cuts
             or native_tts_cancel or media_transport != 'http-paired-v1' or tts_mode != 'native_voxcpm'
             or (attended_visual is False and expected_asr is not None)
             or visual_ancestry_policy != 'conversation-tail-every3-v1')):
        raise ValueError('stable selfie requires visual native HTTPS with exact conversation anchor and no cuts/cancel')
    if type(attended_visual) is not bool:
        raise ValueError('explicit attended visual selection required')
    if attended_visual and (visual_intents is not True or turn_interruptions is not False):
        raise ValueError('attended visual requires visual turns without interruption controls')
    if type(paired_cuts) is not bool or (paired_cuts and not visual_intents):
        raise ValueError('paired cuts require explicit visual HTTPS interruption mode')
    if paired_cuts and not turn_interruptions:
        raise ValueError('paired cuts require the interruption presentation gate')
    if (type(visual_intents) is not bool or (visual_intents and
            (not (turn_interruptions or attended_visual or stable_selfie) or not getattr(runtime, 'visual_intents', False)
             or (visual_ancestry_policy is not None and not stable_selfie) or tts_mode!='native_voxcpm'))):
        raise ValueError('visual intent requires bound HTTPS interruptible or attended runtime with latest-tail ancestry')
    if type(turn_interruptions) is not bool:
        raise ValueError('explicit turn_interruptions boolean required')
    if turn_interruptions and media_transport!='http-paired-v1':
        raise ValueError('turn interruption is explicitly HTTPS paired only')
    if media_transport == 'http-paired-v1':
        if media_adapter is not None:raise ValueError('HTTP mode cannot borrow RTC media socket')
        pump_type=ProducerPump
    elif media_transport == 'webrtc-rtp-v1':
        if media_adapter is None:raise ValueError('RTC requires an owned local media socket; no HTTP fallback')
        from .webrtc_producer import RTCProducerPump
        pump_type=RTCProducerPump
    else:raise ValueError('explicit media transport required')
    if type(overlap_send) is not bool:
        raise ValueError('explicit overlap_send bool required')
    if type(tts_mode) is not str or tts_mode not in ('complete', 'sentence_incremental', 'native_voxcpm'):
        raise ValueError('explicit complete, sentence_incremental or native_voxcpm TTS mode required')
    from .tts_preroll import PCMPrerollPolicy
    if tts_mode == 'native_voxcpm':
        if type(tts_preroll) is not PCMPrerollPolicy:
            raise ValueError('native_voxcpm requires immutable PCMPrerollPolicy')
    elif tts_preroll is not None:
        raise ValueError('tts_preroll is native_voxcpm only')
    if type(ten_minute_test) is not bool or type(hold_listening) is not bool:
        raise ValueError('explicit boolean endurance flags required')
    from .visual_ancestry_policy import CANDIDATE
    if visual_ancestry_policy is not None and (type(visual_ancestry_policy) is not str or visual_ancestry_policy != CANDIDATE):
        raise ValueError('explicit candidate visual ancestry policy required')
    if on_ready is not None and not callable(on_ready):
        raise TypeError('on_ready must be callable')
    if canonical_pcm is None:
        canonical_pcm = canonical_ffmpeg_pcm
    if not callable(canonical_pcm):
        raise TypeError('canonical_pcm must be callable')
    if tts_mode == 'native_voxcpm':
        if not callable(getattr(tts, 'stream', None)):
            raise TypeError('missing interface: ready TTS.stream(text,epoch,deadline)')
    elif not callable(getattr(tts,'synthesize',None)):
        raise TypeError('missing interface: ready TTS.synthesize(text,turn_id,directory,deadline)')
    if not 0 < deadline-time.monotonic() <= (10800 if long_session else 1800):
        raise ValueError('finite worker deadline required')
    if binding.profile_sha256 != role_profile_digest(runtime.quality_profile,'speech'):
        raise ValueError('immutable speech-normalized profile binding mismatch')
    root=Path(root).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('private temporary root must be outside repository')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.stat().st_mode & 0o077:raise ValueError('private worker root required')
    commands=Commands(adapter,binding.session,
        **({'turn_interruptions':True} if turn_interruptions else {}),
        **({'long_session':True} if long_session else {}))
    turn_gate=TurnInterruptionGate() if turn_interruptions else None
    inbox=queue.Queue(maxsize=2); jobs=queue.Queue(maxsize=2)
    stop=threading.Event(); interrupted=threading.Event()
    admission_closed=threading.Event()
    accepted=[0]
    origins=set()
    incremental_streams=[]  # at most two finite, retained sources
    native_receipts=[]  # metadata only, at most two admitted epochs
    cancel_jobs=queue.Queue(maxsize=2)
    terminal_events={}  # one per admitted epoch; verified END or skipped before IO
    cancel_receipts=[]
    coverage_observed=set()
    from .phase_timing import PhaseTiming
    timing = PhaseTiming(component='gpu_worker', emit=True)
    if tts_mode == 'native_voxcpm':
        # Worker-local categories; do not mutate the shared legacy timing contract.
        timing.STAGES = timing.STAGES | frozenset(('native_tts_stream',
            'native_tts_first_eligible_pcm', 'native_tts_protocol_END'))
    if native_tts_cancel:
        timing.STAGES = timing.STAGES | frozenset(('native_tts_cancel_control',
            'native_tts_cancel_terminal', 'native_tts_cancelled_END'))
    from .window_profiler import WindowProfiler
    window_profile = WindowProfiler(window_cap=2542 if long_session else 144)

    class TimedTTS:
        def synthesize(self, *args):
            with timing.stage('tts_sentence'):
                return tts.synthesize(*args)

    def timed_converter(wav, until):
        with timing.stage('canonical_pcm'), window_profile.activity('canonical_pcm'):
            return canonical_pcm(wav, until)
    if expected_asr is not None:
        from research.h3_personal.asr_source_contract import validate_expected_asr
        expected_asr = validate_expected_asr(expected_asr)
    state_lock=threading.Lock()
    control=[None]
    def put(target,value):
        while not stop.is_set() and not interrupted.is_set() and time.monotonic()<deadline:
            if target is not jobs and turn_gate is not None and not turn_gate.current(value['epoch']):return
            try:target.put(value,timeout=.05);return
            except queue.Full:pass
    def interrupt(row):
        with state_lock:
            control[0]=row
            interrupted.set()
        if media_adapter is not None:
            try:media_adapter.shutdown(2)
            except OSError:pass
    def reader():
        try:
            accepted_turns = 0
            last_epoch = 0
            for _ in range(32):
                if stop.is_set():return
                row=commands.receive()
                if stop.is_set():return
                if row['kind']=='stop':
                    admission_closed.set()
                    return
                if row['kind']=='interrupt_turn':
                    if turn_gate is None:raise ValueError('turn interruption not enabled')
                    turn_gate.invalidate(row)
                    timing.record('interrupt_received')
                    if native_tts_cancel:cancel_jobs.put_nowait(row['epoch'])
                    continue
                if row['kind']!='turn':
                    interrupt(row)
                    return
                if accepted_turns >= max_turns:raise ValueError('turn count exceeded')
                reply = validate_turn(row, expected_asr, visual_intents=visual_intents, stable_selfie=stable_selfie)
                if row['epoch'] <= last_epoch:
                    raise ValueError('new turn epoch required')
                last_epoch = row['epoch']
                if tts_mode == 'sentence_incremental':
                    from .relay_incremental_tts import sentences
                    sentences(reply)
                accepted_turns += 1
                accepted[0]=accepted_turns
                if turn_gate is not None:turn_gate.accept(row['epoch'])
                if native_tts_cancel:terminal_events[row['epoch']]=threading.Event()
                if visual_intents:
                    backend.accept_visual_intent(row['epoch'], row['payload']['visual_intent'])
                    print(json.dumps(dict(type='h3_visual_intent_received',session=binding.session,
                        **backend.visual_events[-1]),allow_nan=False),flush=True)
                put(jobs,row)
            raise ValueError('command count exceeded')
        except BaseException:
            if not stop.is_set():interrupt(None)
    def cancel_native():
        # Finite owned control lane, independent of blocked media read/native IO.
        # Never touch the H3 driver or destruct a PCM source from this thread.
        try:
            while not stop.is_set() and not interrupted.is_set():
                try:epoch=cancel_jobs.get(timeout=.05)
                except queue.Empty:continue
                terminal=terminal_events[epoch]
                row=dict(epoch=epoch,clock='gpu_worker_monotonic',
                         started_monotonic=time.monotonic(),status='already_terminal')
                cancel_receipts.append(row)
                if terminal.is_set():continue
                try:
                    with timing.stage('native_tts_cancel_control'):
                        ack=tts.request_cancel(epoch,min(deadline,time.monotonic()+5))
                    row.update(status='requested',ack=ack,ack_monotonic=time.monotonic())
                except Exception:
                    # No HTTP close/retry as fake cancel. A verified ordinary END
                    # within this bounded fallback still permits healthy reuse.
                    row.update(status='bounded_drain',reason='control_unavailable_or_uncertain')
                until=min(deadline,time.monotonic()+NATIVE_CANCEL_TERMINAL_SECONDS)
                while not terminal.wait(.05):
                    if stop.is_set() or interrupted.is_set():return
                    if time.monotonic()>=until:
                        row.update(status='terminal_timeout',finished_monotonic=time.monotonic())
                        interrupt(None)
                        tts.close()  # failed/unknown session, not normal cancellation
                        return
                row['finished_monotonic']=time.monotonic()
                timing.record('native_tts_cancel_terminal')
        except BaseException:
            if not stop.is_set():interrupt(None)

    def synthesize():
        # Sole owner of TTS stdin/stdout/buffer. Never overlap requests or
        # abandon a response then reuse that same child for another request.
        try:
            while not stop.is_set() and not interrupted.is_set() and time.monotonic()<deadline:
                try:row=jobs.get(timeout=.05)
                except queue.Empty:continue
                if stop.is_set() or interrupted.is_set():return
                if turn_gate is not None and not turn_gate.current(row['epoch']):
                    turn_gate.complete(row['epoch'],skipped=True)
                    if native_tts_cancel:terminal_events[row['epoch']].set()
                    continue
                reply=validate_turn(row, expected_asr, visual_intents=visual_intents, stable_selfie=stable_selfie)
                turn_id=f"{'turn' if row['payload']['asr_used'] else 'typed'}-{row['epoch']}"
                if tts_mode == 'native_voxcpm':
                    from .relay_native_tts import produce
                    b=PCMStreamBinding(binding.session,turn_id,row['epoch'],binding.profile_sha256)
                    def publish_native(stream):
                        # Keep the stream alive until END/driver-tail cleanup, even
                        # if a turn interrupt makes its publication stale.
                        timing.record('speech_pcm_ready')
                        timing.record('native_tts_first_eligible_pcm')
                        incremental_streams.append(stream)
                        put(inbox,dict(row,stream=stream))
                    def coverage(stream):
                        if b.epoch in coverage_observed:return
                        from .streaming_windows import inspect_audio_window
                        offer=inspect_audio_window(stream,onset_frame=124,index=0,context=context)
                        if offer.status=='ready':
                            coverage_observed.add(b.epoch)
                            timing.record('h3_initial_coverage_ready')
                            print(json.dumps(dict(type='h3_audio_coverage',clock='gpu_worker_monotonic',
                                observed_monotonic=time.monotonic(),source_epoch=b.epoch,
                                required_source_samples=offer.required_source_samples,
                                available_source_samples=offer.available_source_samples,eos=offer.eos,
                                reason=offer.reason),allow_nan=False),flush=True)
                    with timing.stage('native_tts_stream'):
                        receipt=produce(tts,reply,row['epoch'],b,deadline,publish_native,
                            lambda: not stop.is_set() and not interrupted.is_set(),
                            preroll=tts_preroll,**({'observe':coverage} if paired_cuts else {}),
                            **({'allow_interrupted':True} if native_tts_cancel else {}))
                    if native_tts_cancel:
                        if receipt.stream not in incremental_streams:
                            incremental_streams.append(receipt.stream)  # zero/unpublished prefix also owned
                        if receipt.completion=='interrupted_protocol_END' and turn_gate.current(b.epoch):
                            raise RuntimeError('unexpected cancellation of current source')
                        if receipt.completion=='interrupted_protocol_END':
                            timing.record('native_tts_cancelled_END')
                    from .voxcpm_endpoint_protocol import PINS
                    native_receipts.append(dict(source_id=b.source_id,epoch=b.epoch,
                        pins=dict(PINS),native_sha256=receipt.native_sha256,
                        converted_sha256=receipt.converted_sha256,
                        native_samples=receipt.native_samples,converted_samples=receipt.converted_samples,
                        saturation_count=receipt.saturation_count,native_format=receipt.native_format,
                        converted_format=receipt.converted_format,completion=receipt.completion,
                        watermark_applied=False,model_eos_verified=False,
                        preroll=None if receipt.preroll is None else asdict(receipt.preroll),
                        preroll_policy=asdict(tts_preroll)))
                    timing.record('native_tts_protocol_END')
                    if turn_gate is not None:turn_gate.complete(row['epoch'])
                    if native_tts_cancel:terminal_events[row['epoch']].set()
                    continue
                if tts_mode == 'sentence_incremental':
                    from .relay_incremental_tts import produce
                    b=PCMStreamBinding(binding.session,turn_id,row['epoch'],binding.profile_sha256)
                    def publish(stream):
                        timing.record('speech_pcm_ready')
                        incremental_streams.append(stream)
                        put(inbox,dict(row,stream=stream))
                    produce(TimedTTS(),reply,turn_id,b,root,deadline,timed_converter,
                        complete_pcm,publish,
                        lambda: not stop.is_set() and not interrupted.is_set(),
                        validate_wav=validate_complete_wav)
                    timing.record('tts_eos')
                    if turn_gate is not None:turn_gate.complete(row['epoch'])
                    continue
                with tempfile.TemporaryDirectory(dir=root) as tmp:
                    directory=Path(tmp)/'tts'
                    with timing.stage('tts_complete_wav'), window_profile.activity('tts_complete_wav'):
                        result=tts.synthesize(reply,turn_id,directory,deadline)
                    if stop.is_set() or interrupted.is_set():return
                    pcm=complete_pcm(result,reply,turn_id,directory,timed_converter,deadline)
                timing.record('speech_pcm_ready')
                timing.record('tts_eos')
                put(inbox,dict(row,pcm=pcm))
                if turn_gate is not None:turn_gate.complete(row['epoch'])
        except BaseException:
            if not stop.is_set() and not interrupted.is_set():interrupt(None)
    thread=threading.Thread(target=reader,name='h3-live-command-reader')
    synthesis_thread=threading.Thread(target=synthesize,name='h3-live-tts')
    cancel_thread=(threading.Thread(target=cancel_native,name='h3-live-tts-cancel')
                   if native_tts_cancel else None)
    backend=None;driver=None;success=False
    native_may_be_used=False
    try:
        options = {} if visual_ancestry_policy is None else {'visual_ancestry_policy': visual_ancestry_policy}
        if stable_selfie:options['stable_selfie'] = True
        backend=LiveNativeBackend(runtime,root/'native',**options)
        if stable_selfie and getattr(backend, 'stable_selfie', False) is not True:
            raise ValueError('stable selfie requires bound static native backend')
        driver=_conversation(root/'conversation',binding=binding,context=context,
            deadline=deadline, ten_minute_test=ten_minute_test, long_session=long_session)
        backend._window_profiler = window_profile
        driver._window_profiler = window_profile
        if on_ready is not None:
            on_ready()
        if cancel_thread is not None:cancel_thread.start()
        thread.start()
        synthesis_thread.start()
        turn_count=[0]
        def packets():
            nonlocal native_may_be_used
            turns=0;epoch=0
            speech_sources=set()
            stream_presentation=0
            tail_epoch=None;tail_presentation=0
            quarantine_presentation=None
            quarantine_source_epoch=None
            while True:
                if time.monotonic()>=deadline:raise TimeoutError('finite worker deadline')
                if interrupted.is_set():
                    with state_lock:command=control[0]
                    if command is None:raise RuntimeError('command reader or synthesis failed')
                    raise InterruptedError('finite worker stopped or cancelled')
                if turn_gate is not None:
                    # Reader only invalidates admission. The serial native owner
                    # retires after the preceding step's commit+decoded ACK once.
                    active=driver._stream
                    if active is not None and not turn_gate.current(active.binding.epoch):
                        was_started=driver.control.snapshot()['speech_onset_frame'] is not None
                        driver.request_speech_interrupt(active.binding.epoch)
                        driver.retire_interrupted_speech()
                        timing.record('speech_retired')
                        if was_started:
                            quarantine_presentation=stream_presentation
                            quarantine_source_epoch=active.binding.epoch
                    elif (tail_epoch is not None and not turn_gate.current(tail_epoch)
                          and quarantine_presentation is None):
                        quarantine_presentation=tail_presentation
                        quarantine_source_epoch=tail_epoch
                if driver.inspect().status=='end':driver.retire_speech()
                if turn_gate is None:
                    should_stop=_should_stop_listening(turns, accepted[0], admission_closed.is_set(), hold_listening, max_turns)
                else:
                    view=turn_gate.snapshot()
                    should_stop=(view['synthesis_drained'] and inbox.empty()
                        and (admission_closed.is_set() or (view['accepted_turns']==max_turns and not hold_listening)))
                if should_stop and driver._stream is None and quarantine_presentation is None:
                    driver.request_stop()
                elif driver._stream is None and quarantine_presentation is None:
                    # Bounded inbox contains at most the two accepted turns.
                    while True:
                        try:row=inbox.get_nowait()
                        except queue.Empty:row='waiting'
                        if (row is None or row=='waiting' or turn_gate is None
                                or turn_gate.current(row['epoch'])):break
                    if row is None:raise RuntimeError('command reader failed')
                    if row!='waiting':
                        if row['kind']!='turn':raise InterruptedError('finite worker cancelled')
                        reply=validate_turn(row, expected_asr, visual_intents=visual_intents, stable_selfie=stable_selfie)
                        if (turn_gate is not None and tail_epoch is not None
                                and not turn_gate.current(tail_epoch)):
                            # interrupt+new TTS may arrive after this loop's top
                            # check. Requeue before binding the fresh source so
                            # the old decoder tail cannot share its first packet.
                            inbox.put_nowait(row)
                            quarantine_presentation=tail_presentation
                            quarantine_source_epoch=tail_epoch
                            continue
                        if row['epoch']<=epoch:raise ValueError('new turn epoch required')
                        epoch=row['epoch']
                        turn_id=f"{'turn' if row['payload']['asr_used'] else 'typed'}-{epoch}"
                        if tts_mode in ('sentence_incremental', 'native_voxcpm'):
                            stream=row['stream']
                        else:
                            pcm=row['pcm']
                            b=PCMStreamBinding(binding.session,turn_id,epoch,binding.profile_sha256)
                            stream=PCMIngress(b,timeout=min(300,max(.001,deadline-time.monotonic())))
                            for offset in range(0,len(pcm),64000):
                                stream.append(PCMChunk(b,offset//64000,offset//2,pcm[offset:offset+64000]))
                            stream.seal(total_samples=len(pcm)//2,sha256=hashlib.sha256(pcm).hexdigest())
                        with state_lock:
                            if interrupted.is_set():raise InterruptedError('stale synthesis discarded')
                            if turn_gate is not None and not turn_gate.current(row['epoch']):
                                if tts_mode=='complete':stream.cancel()
                                continue  # Publication raced turn invalidation; never attach stale PCM.
                            driver.attach(stream);turns+=1;turn_count[0]=turns
                            if turn_gate is not None:
                                stream_presentation=turn_gate.snapshot()['presentation_epoch']
                            origins.add(row['payload']['origin'])
                step_epoch=(None if turn_gate is None or driver._stream is None
                            else driver._stream.binding.epoch)
                # Derive from admitted work/tail, not the latest control value:
                # an interrupt can arrive between the loop check and this step.
                packet_presentation=(quarantine_presentation if quarantine_presentation is not None
                    else stream_presentation if step_epoch is not None
                    else tail_presentation if tail_epoch is not None
                    else turn_gate.snapshot()['presentation_epoch'] if turn_gate is not None else 0)
                window_started=time.monotonic()
                previous_receipt=(dict(backend.receipts[-1]) if paired_cuts and quarantine_presentation is not None
                                  and backend.receipts else None)
                cut_view=turn_gate.snapshot() if paired_cuts and quarantine_presentation is not None else None
                cut_frontier=driver.control.snapshot() if cut_view is not None else None
                if quarantine_presentation is not None:timing.record('quarantine_start')
                # From this boundary onward, failure does not prove a local-only
                # fault or a healthy native decoder. Retain its owner on failure.
                native_may_be_used=True
                packet=driver.step(backend)
                if packet is None:
                    if tts_mode not in ('sentence_incremental', 'native_voxcpm'):
                        raise RuntimeError('unexpected unavailable complete source')
                    # Started speech cannot be padded with listening PCM. Wait
                    # cooperatively for real source while checking cancel/deadline.
                    stop.wait(.01)
                    continue
                if paired_cuts and quarantine_presentation is not None:
                    if driver._last_role!='listening':raise ValueError('cut requires actual quarantine listening')
                    packet=_quarantine_suffix(packet,previous_receipt,quarantine_presentation,cut_view,
                        source_epoch=quarantine_source_epoch,frontier=cut_frontier,current=backend.receipts[-1])
                    packet_presentation=packet.envelope.presentation_epoch
                    timing.record('paired_cut_ready')
                    from dataclasses import asdict
                    print(json.dumps(dict(type='h3_paired_cut',sequence=packet.envelope.index,
                        frame_start=packet.envelope.frame_start,frame_stop=packet.envelope.frame_stop,
                        presentation_epoch=packet_presentation,cut=asdict(packet.envelope.presentation_cut),
                        previous_native_committed=previous_receipt['committed_frame'],
                        previous_native_emitted=previous_receipt['emitted_frame'],
                        visual_achievement='unverified'),allow_nan=False),flush=True)
                if visual_intents:
                    backend.receipts[-1]['visual_conditioning']['presentation_epoch']=packet_presentation
                    # Native tail can include generated-but-discarded movement.
                    # Neither submission nor ACK proves a viewer saw the action.
                    from .ordered_metadata import project_metadata
                    metadata=project_metadata(backend.receipts[-1],
                        expected_identity=getattr(getattr(backend,'runtime',None),
                                                  'ordered_reference_sha256',None))
                    event=dict(type='h3_visual_conditioning',session=binding.session,
                        **_visual_window_evidence(backend.receipts[-1], stable_selfie=stable_selfie,
                            anchor_sha256=getattr(backend, '_visual_anchor', None)),
                        **({'ordered_metadata':metadata} if metadata is not None else {}))
                    encoded=json.dumps(event,allow_nan=False)
                    if len(encoded.encode())>65536:raise ValueError('native log line bound')
                    print(encoded,flush=True)
                if turn_gate is not None:
                    if driver._last_role=='speech':
                        tail_epoch=step_epoch;tail_presentation=packet_presentation
                    else:
                        tail_epoch=None
                    if quarantine_presentation is not None:
                        if driver._last_role!='listening':
                            raise RuntimeError('interrupted decoder tail requires one listening quarantine window')
                        quarantine_presentation=None
                        quarantine_source_epoch=None
                    packet=_presentation_packet(packet,packet_presentation)
                timing.record('h3_window', duration=time.monotonic()-window_started)
                window_profile.emit_window(getattr(packet.envelope, 'index', None))
                for region in getattr(packet.envelope, 'regions', ()):
                    if region.kind == 'speech':
                        source = (region.source_id, region.source_epoch)
                        if source not in speech_sources:
                            speech_sources.add(source)
                            timing.record('first_speech_packet_ready')
                yield packet
                if packet.envelope.conversation_final:return
        producer=pump_type(media_adapter if media_adapter is not None else adapter,binding,deadline=deadline,
                              **({'overlap_send': True} if overlap_send else {}))
        try:
            result=producer.run(packets())
        finally:
            _emit_producer_evidence(producer, media_adapter if media_adapter is not None else adapter, failed=sys.exc_info()[0] is not None)
        backend.retire();success=True
        timing.completed=True
        origin=next(iter(origins)) if len(origins)==1 else ('mixed' if origins else None)
        return dict(result,origin=origin,turns=turn_count[0],
            **({'visual_intent_events':backend.visual_events,
                'visual_windows':[_visual_window_evidence(r, stable_selfie=stable_selfie,
                    anchor_sha256=getattr(backend, '_visual_anchor', None)) for r in backend.receipts]}
               if visual_intents else {}),
            **({'turn_interruption':turn_gate.snapshot()} if turn_gate is not None else {}),
            **({'native_tts_receipts':native_receipts} if tts_mode=='native_voxcpm' else {}),
            **({'native_tts_cancel_receipts':cancel_receipts} if native_tts_cancel else {}),
            tts_availability=(
            'native-voxcpm-no-watermark-float32-to-explicit-S16LE-protocol-END-not-model-EOS'
            if tts_mode == 'native_voxcpm' else
            'sentence-incremental-watermarked-WAV' if tts_mode == 'sentence_incremental'
            else 'complete-watermarked-WAV'))
    finally:
        primary=sys.exc_info()[1]
        if media_adapter is not None:
            try:media_adapter.shutdown(2)
            except OSError:pass
            try:media_adapter.close()
            except OSError:pass
        cleanup_errors=[]
        def cleanup(action):
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
        stop.set();cleanup(adapter.close)
        if thread.ident is not None:cleanup(lambda: thread.join(timeout=2))
        if synthesis_thread.ident is not None:
            cleanup(lambda: synthesis_thread.join(timeout=2))
            if synthesis_thread.is_alive():
                # Closing the child unblocks its sole reader; never reuse it.
                close=getattr(tts,'close',None)
                if callable(close):cleanup(close)
                cleanup(lambda: synthesis_thread.join(timeout=2))
        if cancel_thread is not None and cancel_thread.ident is not None:
            cleanup(lambda: cancel_thread.join(timeout=6))
            if cancel_thread.is_alive():
                cleanup(tts.close)
                cleanup(lambda: cancel_thread.join(timeout=1))
        cleanup(commands.close)
        if native_tts_cancel and cancel_thread is not None and not cancel_thread.is_alive():
            cleanup(lambda: print(json.dumps(dict(type='h3_native_tts_cancel_receipts',
                receipts=cancel_receipts[:2]),allow_nan=False),flush=True))
        if tts_mode=='native_voxcpm' and not synthesis_thread.is_alive():
            cleanup(lambda: (root/'native-tts-receipts.json').write_text(
                json.dumps(native_receipts,allow_nan=False,indent=2)))
            # Only completed protocol receipts, after the producer has stopped.
            # Keep stdout bounded and metadata-only, even on a failed run.
            cleanup(lambda: print(json.dumps(dict(type='h3_native_tts_receipts',
                receipts=native_receipts[:2]),allow_nan=False),flush=True))
        for stream in incremental_streams:cleanup(stream.cancel)
        if driver is not None:cleanup(driver.close)
        if backend is not None and not success and not native_may_be_used:
            cleanup(backend.close_unused)
        # No supported local abandonment reason can be established after step
        # (including retire/ACK uncertainty). Leave the native owner attached;
        # the backend itself quarantines native failures. Never claim recovery.
        if thread.is_alive() or synthesis_thread.is_alive() or (cancel_thread is not None and cancel_thread.is_alive()):
            cleanup_errors.append(RuntimeError('worker threads did not terminate'))
        window_profile.emit_remaining()
        # Metadata-only stdout events survive an abrupt provider cancellation;
        # this final private receipt is best-effort and never masks that failure.
        try:
            timing.persist(root)
        except Exception:
            cleanup(lambda: print('{"type":"h3_stage_timing_receipt","status":"unavailable"}', flush=True))
        if primary is None and cleanup_errors:
            raise cleanup_errors[0]
