"""Finite 1024 listening→speech→listening scenario contract (CPU only).

Three immutable sources on ONE conversation clock:

    listening [0,124) → speech [124,430) → listening [430,532)

Every source keeps its own Session and canonical audio; the conversation owner
carries the visual tail and the persistent decoder across the boundaries. This
module only builds manifests in the required order, validates the actual
emitted-frame sequence (committed F versus emitted F-5) and feeds the single
conversation AV sink. It never launches or reloads a model.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import wave

from .conversation_timeline import ConversationBinding, FrozenPCMSource, TimelineSpan
from .conversation_sequence import ConversationSequence
from .continuation_candidate import make_options
from .continuation_plan import Window, audio_boundary, plan as continuation_plan, video_t


SCHEMA = 'h3-conversation-scenario-v1'
FPS = Fraction(24)
WINDOW_FRAMES = 124
OVERLAP_FRAMES = 22
TOTAL_FRAMES = 532
ONSETS = (0, 124, 430)
ROLES = ('listening', 'speech', 'listening')
HELD_FRAMES = 5
CANDIDATE = make_options(window_audio_context=True)
SEED = 314159


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('strict integer out of range')


def _digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('lowercase SHA256 required')


def _id(value):
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError('bounded explicit ID required')


@dataclass(frozen=True)
class SourcePlan:
    epoch: int
    role: str
    source_id: str
    wav: Path
    wav_sha256: str
    samples: int
    sample_rate: int
    channels: int
    sample_width: int
    onset_frame: int
    total_frames: int
    conversation_final: bool

    def __post_init__(self):
        _integer(self.epoch)
        _integer(self.samples, 1)
        _integer(self.sample_rate, 1)
        if self.role not in ('listening', 'speech'):
            raise ValueError('declared public role required')
        _id(self.source_id)
        _digest(self.wav_sha256)
        if (self.epoch, self.role, self.onset_frame) != (
                self.epoch, ROLES[self.epoch], ONSETS[self.epoch]):
            raise ValueError('fixed role/onset schedule required')
        if self.total_frames != (WINDOW_FRAMES, 328, 124)[self.epoch]:
            raise ValueError('fixed source-local frame total required')
        if type(self.conversation_final) is not bool or self.conversation_final != (self.epoch == 2):
            raise ValueError('only the last source is conversation-final')

    @property
    def windows(self):
        return continuation_plan(self.total_frames, WINDOW_FRAMES, OVERLAP_FRAMES)

    @property
    def committed_endpoints(self):
        base = self.onset_frame - OVERLAP_FRAMES if self.epoch else 0
        return tuple(base + w.end for w in self.windows)

    @property
    def guide_start(self):
        return self.onset_frame - OVERLAP_FRAMES if self.epoch else None


@dataclass(frozen=True)
class ScenarioPlan:
    conversation_id: str
    profile_sha256: str
    sources: tuple[SourcePlan, ...]

    def __post_init__(self):
        _id(self.conversation_id)
        _digest(self.profile_sha256)
        if type(self.sources) is not tuple or len(self.sources) != 3:
            raise ValueError('exactly three sources required')
        if [s.epoch for s in self.sources] != [0, 1, 2]:
            raise ValueError('ordered finite epochs required')
        if len({s.source_id for s in self.sources}) != 3:
            raise ValueError('unique source IDs required')
        if (self.sources[0].sample_rate, self.sources[0].channels) != (
                self.sources[1].sample_rate, self.sources[1].channels):
            raise ValueError('no resampling or channel fallback')
        cursor = ONSETS[0]
        for source in self.sources:
            if source.onset_frame != cursor:
                raise ValueError('contiguous source onsets required')
            cursor = source.committed_endpoints[-1]
        if cursor != TOTAL_FRAMES:
            raise ValueError('conversation must end at the fixed 532-frame bound')

    @property
    def frames(self):
        return TOTAL_FRAMES

    @property
    def committed_endpoints(self):
        return tuple(e for s in self.sources for e in s.committed_endpoints)

    @property
    def emitted_endpoints(self):
        final = self.committed_endpoints[-1]
        return tuple(final if e == final else e - HELD_FRAMES for e in self.committed_endpoints)

    @property
    def segments(self):
        cursor = 0
        out = []
        for stop in self.emitted_endpoints:
            out.append((cursor, stop))
            cursor = stop
        return tuple(out)

    @property
    def spans(self):
        return (TimelineSpan(0, ONSETS[1], 'listening'),
                TimelineSpan(ONSETS[1], ONSETS[2], 'speech', self.sources[1].source_id),
                TimelineSpan(ONSETS[2], TOTAL_FRAMES, 'listening'))

    def binding(self):
        speech = self.sources[1]
        raw = Path(speech.wav).read_bytes()
        if sha256(raw).hexdigest() != speech.wav_sha256:
            raise ValueError('speech WAV changed')
        with wave.open(io.BytesIO(raw), 'rb') as handle:
            if (handle.getcomptype(), handle.getsampwidth(), handle.getframerate(),
                    handle.getnchannels(), handle.getnframes()) != (
                    'NONE', speech.sample_width, speech.sample_rate, speech.channels, speech.samples):
                raise ValueError('speech WAV metadata differs')
            pcm = handle.readframes(speech.samples)
            if len(pcm) != speech.samples * speech.channels * speech.sample_width:
                raise ValueError('truncated original speech PCM')
        source = FrozenPCMSource(speech.source_id, pcm, speech.sample_rate,
                                 speech.channels, sha256(pcm).hexdigest())
        return ConversationBinding(self.conversation_id, (source,), self.spans,
                                   speech.sample_rate, speech.channels, FPS)

    def manifests(self):
        """Exact per-window manifests in the only accepted generation order."""
        out = []
        for source in self.sources:
            windows = source.windows
            for index, window in enumerate(windows):
                final = index == len(windows) - 1
                manifest = dict(
                    schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
                    wav=str(source.wav), sha256=source.wav_sha256,
                    source_samples=source.samples, source_sample_rate=source.sample_rate,
                    total_frames=source.total_frames, window_frames=WINDOW_FRAMES,
                    overlap_frames=OVERLAP_FRAMES, expected_windows=len(windows),
                    seed=SEED, sampler='SMD4', fps=24,
                    global_audio_t=audio_boundary(source.total_frames),
                    audio_capacity_samples_32000=audio_boundary(source.total_frames) * 800,
                    session_id=source.source_id, index=index, final=final,
                    candidate=dict(CANDIDATE),
                    conversation=dict(schema='h3-model-conversation-v1',
                        id=self.conversation_id, profile_sha256=self.profile_sha256,
                        source_epoch=source.epoch, source_id=source.source_id,
                        role=source.role, source_onset_frame=source.onset_frame,
                        conversation_final=source.conversation_final and final,
                        total_frames=TOTAL_FRAMES),
                    source_index=index)
                out.append((source.epoch, index, manifest))
        return tuple(out)

    def window_emissions(self):
        """(source_epoch, index, emitted frames, conversation_final) sequence."""
        out = []
        cursor = 0
        for source in self.sources:
            for index, endpoint in enumerate(source.committed_endpoints):
                final = source.conversation_final and index == len(source.committed_endpoints) - 1
                stop = endpoint if final else endpoint - HELD_FRAMES
                out.append((source.epoch, index, stop - cursor, final))
                cursor = stop
        return tuple(out)


def plan(*, conversation_id, profile_sha256, listening_wavs, speech_wav):
    """Build the fixed scenario from explicit staged silence and speech WAVs."""
    if type(listening_wavs) is not tuple or len(listening_wavs) != 2:
        raise ValueError('two explicit encoded-silence WAVs required')
    speech = Path(speech_wav).resolve(strict=True)
    source_plans = []
    staged = ((0, 'listening', listening_wavs[0]), (1, 'speech', speech),
              (2, 'listening', listening_wavs[1]))
    for epoch, role, path in staged:
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        if not 0 < len(raw) <= 32 * 1024 * 1024:
            raise ValueError('bounded staged WAV required')
        with wave.open(io.BytesIO(raw), 'rb') as handle:
            channels, width, rate, samples = (handle.getnchannels(), handle.getsampwidth(),
                                              handle.getframerate(), handle.getnframes())
            if (handle.getcomptype() != 'NONE' or width != 2 or channels not in (1, 2)
                    or not 0 < samples <= rate * 60):
                raise ValueError('original S16LE PCM WAV required')
            pcm = handle.readframes(samples)
            if len(pcm) != samples * channels * width:
                raise ValueError('truncated staged PCM')
        if role == 'listening':
            # Explicit silence, never an omitted condition or a zero latent.
            if any(pcm):
                raise ValueError('listening source must be explicit encoded silence')
        else:
            capacity = audio_boundary(328) * 800 * rate
            span = (ONSETS[2] - ONSETS[1]) * rate * FPS.denominator // FPS.numerator
            if samples * 32000 > capacity:
                raise ValueError('speech exceeds audio capacity; never truncate or stretch')
            if samples > span:
                raise ValueError('speech does not fit its declared span; never shorten it')
        source_plans.append(SourcePlan(epoch, role, f'conversation-{epoch}',
            path, sha256(raw).hexdigest(), samples, rate, channels, width,
            ONSETS[epoch], (WINDOW_FRAMES, 328, 124)[epoch], epoch == 2))
    return ScenarioPlan(conversation_id, profile_sha256, tuple(source_plans))


def segment_timing(index, frame_start, frame_stop):
    from .realtime_segments import SegmentTiming
    _integer(index)
    _integer(frame_start)
    _integer(frame_stop, frame_start + 1)
    return SegmentTiming(index, 0, frame_stop - frame_start, frame_start, frame_stop)


def read_window_receipts(coverage, scenario, epoch, index):
    """Validate both-rank conversation receipts for one committed window.

    Returns evidence usable as a handoff receipt. A single rank or a mismatched
    commit/source/canonical encode is a hard failure, never a self-declared pass.
    """
    coverage = Path(coverage).resolve(strict=True)
    source = scenario.sources[epoch]
    committed = source.committed_endpoints[index]
    rows, raw = [], []
    for rank in (0, 1):
        path = coverage / f'h3-conversation-rank-{rank}-epoch-{epoch}-window-{index}.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
            raise ValueError(f'conversation rank receipt missing/oversized: rank {rank}')
        raw.append(path.read_bytes())
        row = json.loads(raw[-1])
        if not isinstance(row, dict) or set(row) != {
                'schema', 'conversation', 'committed_frames', 'source_id',
                'canonical_audio_sha256', 'guide_digests', 'audio_origin'}:
            raise ValueError('exact conversation receipt fields required')
        conversation = row['conversation']
        if (row['schema'] != 'h3-model-conversation-v1'
                or conversation.get('id') != scenario.conversation_id
                or conversation.get('source_epoch') != epoch
                or conversation.get('source_id') != source.source_id
                or conversation.get('role') != source.role
                or conversation.get('source_onset_frame') != source.onset_frame
                or conversation.get('total_frames') != TOTAL_FRAMES
                or conversation.get('profile_sha256') != scenario.profile_sha256
                or row['committed_frames'] != committed
                or row['source_id'] != source.source_id):
            raise ValueError('conversation receipt commit/source/role drift')
        from .conversation_audio import audio_origin
        if row['audio_origin'] != audio_origin(source.onset_frame, source.total_frames):
            raise ValueError('conversation waveform origin/preroll drift')
        _digest(row['canonical_audio_sha256'])
        guides = row['guide_digests']
        if type(guides) is not list or len(guides) != 3 or any(
                g is not None and (type(g) is not str or not re.fullmatch('[0-9a-f]{64}', g))
                for g in guides):
            raise ValueError('bounded guide digests required')
        rows.append(row)
    if (rows[0]['canonical_audio_sha256'] != rows[1]['canonical_audio_sha256']
            or rows[0]['guide_digests'] != rows[1]['guide_digests']
            or rows[0]['committed_frames'] != rows[1]['committed_frames']):
        raise ValueError('both-rank conversation receipt disagreement')
    return dict(committed_frames=committed, source_id=source.source_id,
                canonical_audio_sha256=rows[0]['canonical_audio_sha256'],
                audio_origin=dict(rows[0]['audio_origin']),
                guide_digests=list(rows[0]['guide_digests']),
                receipt_digest=sha256(b'\x00'.join(raw)).hexdigest(),
                ranks=rows)


class ConversationRenderer:
    """Drive one finite scenario: manifests in order, then one AV sink segment.

    backend.render_window(source_epoch, index, final) must return dict(raw,
    count, interval). raw is exported RGB at (count, 448, 252, 3); count is the
    ACTUAL emitted frame count (committed-5 except at the conversation flush).
    receipts(source_epoch, index) must return validated both-rank receipts.
    """
    def __init__(self, scenario, binding, stream, turn, backend, *, width=252,
                 height=448, manifest_path=None, receipts=None, sequence=None,
                 before_window=None):
        if type(scenario) is not ScenarioPlan:
            raise ValueError('ScenarioPlan required')
        if type(binding) is not ConversationBinding or binding.conversation_id != scenario.conversation_id:
            raise ValueError('scenario conversation binding required')
        if not callable(getattr(backend, 'render_window', None)):
            raise ValueError('explicit render_window backend required')
        if receipts is not None and not callable(receipts):
            raise ValueError('callable receipt reader required')
        if before_window is not None and not callable(before_window):
            raise ValueError('callable before_window hook required')
        if (sequence is not None and type(sequence) is not ConversationSequence):
            raise ValueError('ConversationSequence required')
        if (width, height) != (252, 448):
            raise ValueError('fixed 252x448 exported geometry required')
        self.scenario = scenario
        self.binding = binding
        self.stream = stream
        self.turn = Path(turn).resolve(strict=True)
        self.backend = backend
        self.width, self.height = width, height
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._receipts = receipts
        self._before = before_window
        self.sequence = sequence
        self.receipts = []

    def _publish(self, manifest):
        if self.manifest_path is None:
            return
        temporary = self.manifest_path.with_suffix('.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(manifest, stream, sort_keys=True)
        os.replace(temporary, self.manifest_path)

    def run(self):
        expected = self.scenario.window_emissions()
        plans = {(epoch, index): manifest for epoch, index, manifest in self.scenario.manifests()}
        cursor = 0
        windows = [(s.epoch, len(s.windows)) for s in self.scenario.sources]
        previous = None
        for index, (epoch, window_index, count, final) in enumerate(expected):
            manifest = plans[(epoch, window_index)]
            source_last = window_index == windows[epoch][1] - 1
            if self._before is not None:
                self._before(epoch, window_index)
            if self.sequence is not None:
                if index == 0:
                    self.sequence.listen(handoff_frame=0, visual_receipt=None,
                                         audio_guide_reset_receipt=None)
                elif epoch == 1 and window_index == 0:
                    self.sequence.request_speech()
            self._publish(manifest)
            result = self.backend.render_window(epoch, window_index, final)
            if type(result) is not dict or set(result) != {'raw', 'count', 'interval'}:
                raise ValueError('explicit raw/count/interval result required')
            if type(result['count']) is not int or result['count'] != count:
                raise ValueError('actual emitted frame count differs from F/F-5 plan')
            raw = result['raw']
            if type(raw) is not bytes or len(raw) != count * self.width * self.height * 3:
                raise ValueError('exported RGB byte count differs')
            interval = result['interval']
            if (type(interval) is not tuple or len(interval) != 2
                    or any(type(v) not in (int, float) for v in interval)
                    or interval[0] < 0 or interval[1] < interval[0]):
                raise ValueError('invalid bounded generation interval')
            evidence = None
            if self._receipts is not None:
                evidence = self._receipts(epoch, window_index)
                committed = self.scenario.sources[epoch].committed_endpoints[window_index]
                if (evidence['committed_frames'] != committed
                        or evidence['source_id'] != self.scenario.sources[epoch].source_id):
                    raise ValueError('receipt committed endpoint/source differs')
            if self.sequence is not None and epoch == 1 and window_index == 0:
                if evidence is None:
                    raise ValueError('speech preparation requires both-rank evidence')
                self.sequence.speech_prepared(
                    handoff_frame=cursor, visual_receipt=previous,
                    audio_guide_reset_receipt=evidence['canonical_audio_sha256'])
            if (self.sequence is not None and epoch == 2 and window_index == 0):
                if evidence is None:
                    raise ValueError('listening handoff requires both-rank evidence')
                self.sequence.listen(
                    handoff_frame=cursor, visual_receipt=previous,
                    audio_guide_reset_receipt=evidence['canonical_audio_sha256'])
            segment = self.turn / f'conversation-{index:02d}.mp4'
            from research.run_h3_realtime_first import encode_rgb_segment
            encode_rgb_segment(raw, segment, segment_timing(index, cursor, cursor + count),
                               width=self.width, height=self.height)
            from .live_voice_runtime import _metadata
            _metadata(segment, count)
            segment.chmod(0o400)
            event = dict(path=str(segment), index=index, frame_count=count,
                         frame_start=cursor, frame_stop=cursor + count,
                         time_base_numerator=1, time_base_denominator=24,
                         sha256=sha256(segment.read_bytes()).hexdigest(),
                         video_only=True, committed=True, conversation_final=final)
            if self.sequence is not None:
                self.sequence.emit(index=index, frames=count, emitted_end=cursor + count)
            self.stream.accept(event)
            self.receipts.append(dict(segment=str(segment.name), source_epoch=epoch,
                window_index=window_index, frame_start=cursor, frame_stop=cursor + count,
                conversation_final=final, generation_interval=list(interval),
                committed_frames=None if evidence is None else evidence['committed_frames'],
                canonical_audio_sha256=None if evidence is None else evidence['canonical_audio_sha256'],
                audio_origin=None if evidence is None else evidence['audio_origin'],
                receipt_digest=None if evidence is None else evidence['receipt_digest']))
            previous = None if evidence is None else evidence['receipt_digest']
            cursor += count
        if cursor != TOTAL_FRAMES:
            raise ValueError('conversation emitted frames do not cover 532')
        if self.sequence is not None:
            self.sequence.close()
        output = self.stream.finish(conversation_final=True)
        return dict(schema=SCHEMA, output=str(output),
                    conversation_id=self.scenario.conversation_id,
                    profile_sha256=self.scenario.profile_sha256,
                    frames=TOTAL_FRAMES,
                    committed_endpoints=list(self.scenario.committed_endpoints),
                    emitted_endpoints=list(self.scenario.emitted_endpoints),
                    segments=self.receipts, av=self.stream.receipt(),
                    source_onsets=[s.onset_frame for s in self.scenario.sources],
                    roles=[s.role for s in self.scenario.sources],
                    sequence=None if self.sequence is None else self.sequence.receipt(),
                    listening_quality_verified=False, lip_sync_verified=False,
                    device_playback_verified=False, realtime_complete=False)
