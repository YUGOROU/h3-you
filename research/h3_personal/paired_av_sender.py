"""CPU-only serialized credit owner. Reserve BEFORE writing; never retry admitted AV.

Transport exceptions (including partial writes or ambiguous ACK reads) must call
failed() or disconnect(). This module neither executes IO nor judges model health.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import time

from .paired_av import (AVEnvelope, LiveAVBinding, VerifiedReceipt,
                        ProtocolError, CreditError)


@dataclass(frozen=True, eq=False)
class Ticket:
    """Identity capability; only the exact instance issued by reserve is valid."""
    pass


class Sender:
    def __init__(self, binding, deadline, clock=time.monotonic):
        if type(binding) is not LiveAVBinding:
            raise ProtocolError('binding required')
        binding.__post_init__()
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ProtocolError('finite absolute deadline required')
        self._binding = deepcopy(binding)
        self._deadline, self._clock = deadline, clock
        self._last = None
        self._terminal = None
        self._pending = []
        self._bytes = self._index = self._frame = self._acked = self._verified = self._discarded = self._partial_acked = 0
        self._final = False
        self._sources = {}
        self._active = None
        self._epoch = -1
        self._check()

    @property
    def binding(self):
        return deepcopy(self._binding)

    def _stop(self, reason):
        self._terminal = reason
        self._pending.clear()
        self._sources.clear()
        self._active = None
        self._bytes = 0

    def _check(self):
        if self._terminal is not None:
            raise ProtocolError(self._terminal)
        try:
            now = self._clock()
            if (type(now) not in (int, float) or not math.isfinite(now)
                    or (self._last is not None and now < self._last)
                    or now >= self._deadline):
                raise ProtocolError('expired, nonfinite or rewound monotonic clock')
            self._last = now
        except BaseException:
            self._stop('invalid clock or deadline')
            raise

    def reserve(self, envelope):
        self._check()
        try:
            if type(envelope) is not AVEnvelope:
                raise ProtocolError('envelope required')
            envelope = deepcopy(envelope)
            envelope.__post_init__()
            if envelope.binding != self._binding or self._final:
                raise ProtocolError('binding mismatch or already final')
            if envelope.index != self._index or envelope.frame_start != self._frame:
                raise ProtocolError('duplicate, replay or nonsequential segment')
            if envelope.index == self._binding.max_segments - 1 and not envelope.conversation_final:
                raise ProtocolError('last segment must be final')
            if envelope.frame_stop == self._binding.max_total_frames and not envelope.conversation_final:
                raise ProtocolError('last frame must be final')
            sources = dict(self._sources)
            active, epoch = self._active, self._epoch
            for r in envelope.regions:
                if r.kind != 'speech':
                    continue
                key = (r.source_id, r.source_epoch)
                if key != active:
                    if key in sources or r.source_epoch <= epoch or r.source_offset != 0:
                        raise ProtocolError('retired or stale source')
                    active, epoch = key, r.source_epoch
                if key in sources and sources[key] != r.source_offset:
                    raise ProtocolError('source replay or gap')
                sources[key] = r.source_offset + r.sample_stop - r.sample_start
            size = envelope.video_size + envelope.pcm_size
            if len(self._pending) >= 2 or self._bytes + size > self._binding.max_inflight_bytes:
                self._check()
                raise CreditError('sender credits exhausted; not admitted')
            ticket = Ticket()
            self._pending.append([ticket, envelope, False])
            self._bytes += size
            self._index += 1
            self._frame = envelope.frame_stop
            self._final = envelope.conversation_final
            self._sources, self._active, self._epoch = sources, active, epoch
            self._check()
            return ticket
        except CreditError:
            raise
        except BaseException:
            self._stop('reservation protocol failure; no replay')
            raise

    def _entry(self, reference):
        for entry in self._pending:
            if reference is entry[0]:
                return entry
            if type(reference) is VerifiedReceipt:
                reference.envelope.__post_init__()
                if reference.envelope == entry[1]:
                    return entry
        raise ProtocolError('unknown capability or mismatched full receipt')

    def _transition(self, reference, sink, certainty=True, *, discard=False, partial=False):
        self._check()
        try:
            entry = self._entry(reference)
            if sink:
                if certainty is not True or not entry[2] or entry is not self._pending[0]:
                    raise ProtocolError('ambiguous, unverified or non-FIFO disposition')
                if not discard and ((entry[1].presentation_cut is not None) != partial):
                    raise ProtocolError('cut requires an honest partial disposition')
                self._pending.pop(0)
                self._bytes -= entry[1].video_size + entry[1].pcm_size
                if partial:
                    self._partial_acked += 1
                elif discard:
                    self._discarded += 1
                else:
                    self._acked += 1
            else:
                if entry[2]:
                    raise ProtocolError('duplicate verification')
                entry[2] = True
                self._verified += 1
            self._check()
        except BaseException:
            self._stop('acknowledgement protocol failure; no replay')
            raise

    def verified(self, ticket_or_receipt):
        self._transition(ticket_or_receipt, False)

    def sink_ack(self, ticket_or_receipt, *, certainty):
        self._transition(ticket_or_receipt, True, certainty)

    def partial_ack(self, ticket_or_receipt, *, certainty):
        """Settle full verified FIFO credit, not full packet presentation."""
        self._transition(ticket_or_receipt, True, certainty, partial=True)

    def discard_ack(self, ticket_or_receipt, *, certainty):
        """Release verified FIFO credit after an explicit interrupted discard.

        This is distinct from sink acceptance. It cannot infer which epoch to
        discard, release unverified bytes, skip reservations or enable replay.
        """
        self._transition(ticket_or_receipt, True, certainty, discard=True)

    def ack(self, message):
        """Consume the exact strict primitive dict emitted by paired_av_wire."""
        self._check()
        try:
            from .paired_av_wire import parse_ack_message, ack_record
            message = dict(parse_ack_message(message))
            entry = next((e for e in self._pending if e[1].index == message['index']), None)
            if entry is None:
                raise ProtocolError('unknown ACK index')
            e = entry[1]
            expected = ack_record(e,message['stage'])
            if message != expected:
                raise ProtocolError('mismatched ACK')
            self._transition(entry[0], message['stage'] != 'verified',
                             discard=message['stage'] == 'discarded-interrupted',
                             partial=message['stage'] == 'partially-presented-interrupted')
            self._check()
        except BaseException:
            self._stop('wire ACK failure; no replay')
            raise

    def failed(self, ticket):
        """Any uncertain send is terminal, even if the ticket is invalid."""
        self._stop('send failure or ambiguity; no replay')

    def disconnect(self):
        self._stop('disconnected; no replay')

    def close(self):
        self._stop('closed')

    cancel = close

    def snapshot(self):
        if self._terminal is None:
            self._check()
        result = dict(reserved=self._index, verified=self._verified, acked=self._acked,
                      discarded=self._discarded, partial_acked=self._partial_acked,
                      disposed=self._acked+self._discarded+self._partial_acked,
                      pending=len(self._pending), inflight_bytes=self._bytes,
                      frame_stop=self._frame, final=self._final,
                      transport_complete=(self._final and self._acked+self._discarded+self._partial_acked == self._index
                                          and self._terminal is None),
                      presentation_complete=(self._final and self._acked == self._index
                                             and self._terminal is None),
                      terminal=self._terminal, deadline=self._deadline)
        if self._terminal is None:
            self._check()
        return result
