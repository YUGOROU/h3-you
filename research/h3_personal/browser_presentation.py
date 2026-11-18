"""Per-turn presentation-complete evidence for the local HTTPS browser route.

The half-duplex capture gate (``half_duplex_gate.HalfDuplexCaptureGate``) only
accepts sink-level evidence that the *previous* turn's media actually finished
presenting. This module produces that record for the existing loopback browser
transport, from two strictly separated inputs:

* an owner-side declaration that every AV submission for turn ``epoch`` has been
  accepted into the HLS stream (the submitted media end, in media seconds), and
* client-reported playback position (``media_seconds``/``current_time``) values
  that the server already validates as bounded finite scalars.

A record is emitted only when the reported playhead has reached the end of the
media submitted for that epoch. Relay ACKs, worker completion, HTTP HEAD/GET
traffic, playlist length alone, ``first_video``/``first_append`` and synthetic VAD
are never treated as presentation evidence here.

What this does NOT prove (never claim it):

* it is not acoustic/SPL or perceived-completion proof; it is the media pipeline
  having advanced past the turn's media,
* a nonzero encoder PTS origin would shift the playhead relative to the
  frame-based submitted end; the browser route starts its span at frame zero and
  the sink refuses a discontinuous origin when this evidence is enabled,
* the current client reports a playhead only on ``playing``/``waiting``/
  ``stalled`` and terminal events, so a stall-free stream can produce too few
  reports; absence of evidence fails closed (no capture) rather than guessing.

No device, browser, network or GPU ownership; no thread is started here. Waits
are bounded by explicit deadlines and never extend a session.
"""
from __future__ import annotations

import math
import threading
import time

from .half_duplex_gate import EVIDENCE_SCHEMA

SNAPSHOT_SCHEMA = 'h3-browser-presentation-evidence-v1'
MAX_MEDIA_SECONDS = 2400.0
PLAYHEAD_SLACK_SECONDS = 1.0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class BrowserPresentationEvidence:
    """Bounded, monotonic per-turn presentation accounting for one session.

    ``session`` must match the binding the capture gate uses. ``deadline`` is the
    absolute ``time.monotonic()`` conversation deadline. ``max_turns`` bounds how
    many turn epochs may ever be declared.
    """

    def __init__(self, *, session, deadline, max_turns=8,
                 max_media_seconds=MAX_MEDIA_SECONDS):
        if type(session) is not str or not session or len(session) > 128:
            raise ValueError('bounded session identifier required')
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= time.monotonic()):
            raise ValueError('finite monotonic deadline in the future required')
        if type(max_turns) is not int or isinstance(max_turns, bool) or not 1 <= max_turns <= 8:
            raise ValueError('bounded finite turn limit required')
        if (not _finite(max_media_seconds) or not 0 < max_media_seconds <= 1_000_000):
            raise ValueError('bounded media seconds limit required')
        self.session = session
        self.deadline = float(deadline)
        self.max_turns = max_turns
        self.hard_deadline = self.deadline
        self.max_media_seconds = float(max_media_seconds)
        self._condition = threading.Condition(threading.RLock())
        self._ends: list[float] = []
        self._records: list[dict] = []
        self._playhead = 0.0
        self._playhead_reports = 0
        self._rejections = 0
        self._reason = None
        self._stopped = False

    # -- helpers (call with the lock held) ---------------------------------
    def _expire(self):
        if not self._stopped and time.monotonic() >= self.deadline:
            self._stopped, self._reason = True, 'deadline'

    def _fail(self, reason, message):
        self._stopped = True
        self._reason = reason
        self._condition.notify_all()
        raise ValueError(f'presentation evidence stopped: {reason} ({message})')

    def _reject(self, message):
        self._rejections += 1
        raise ValueError(message)

    def _completed_epoch(self, epoch):
        for record in self._records:
            if record['epoch'] == epoch:
                return record
        return None

    # -- owner-side submission declarations --------------------------------
    def note_submitted(self, epoch, media_seconds):
        """Declare that all media for ``epoch`` has been accepted by the sink.

        ``media_seconds`` is the submitted media end on the stream timeline, taken
        from the sink's own accepted frame counter. Epochs must be strictly
        sequential and the end must strictly grow; duplicates are rejected.
        """
        if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
            with self._condition:
                self._reject('bounded turn epoch required')
        if not _finite(media_seconds) or not 0 < media_seconds <= self.max_media_seconds:
            with self._condition:
                self._reject('bounded submitted media end required')
        with self._condition:
            self._expire()
            if self._stopped:
                self._reject('submission after terminal state')
            if epoch != len(self._ends) + 1:
                self._reject('strictly sequential turn submission required')
            if epoch > self.max_turns:
                self._reject('turn submission exceeds finite bound')
            if self._ends and media_seconds <= self._ends[-1]:
                self._reject('submitted media end must strictly grow')
            self._ends.append(float(media_seconds))

    # -- client-reported playhead ------------------------------------------
    def note_playhead(self, media_seconds, *, now=None):
        """Account one client playback position; return any new records.

        Values must be finite, non-negative, non-decreasing and not beyond the
        media that was actually submitted. Records are emitted once, in epoch
        order, and only for media the playhead has already passed.
        """
        if not _finite(media_seconds) or media_seconds < 0:
            with self._condition:
                self._reject('finite non-negative playhead required')
        stamp = time.monotonic() if now is None else now
        if not _finite(stamp) or stamp <= 0:
            with self._condition:
                self._reject('finite playhead timestamp required')
        with self._condition:
            self._expire()
            if self._stopped:
                self._reject('playhead after terminal state')
            if media_seconds > self.max_media_seconds:
                self._reject('playhead exceeds finite media bound')
            if self._ends and media_seconds > self._ends[-1] + PLAYHEAD_SLACK_SECONDS:
                self._reject('playhead beyond submitted media')
            if media_seconds < self._playhead:
                self._fail('playhead_regression', 'client playhead moved backwards')
            self._playhead = float(media_seconds)
            self._playhead_reports += 1
            new = []
            done = len(self._records)
            previous_end = self._ends[done - 1] if done else 0.0
            for offset in range(done, len(self._ends)):
                index, end = offset + 1, self._ends[offset]
                if self._playhead < end:
                    break
                record = dict(schema=EVIDENCE_SCHEMA, session=self.session, epoch=index,
                              completed_at_monotonic=float(stamp),
                              output_seconds=float(end - previous_end))
                self._records.append(record)
                new.append(dict(record))
                previous_end = end
            if new:
                self._condition.notify_all()
            return tuple(new)

    # -- read/wait ---------------------------------------------------------
    def record(self, epoch):
        if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
            raise ValueError('bounded turn epoch required')
        with self._condition:
            return self._completed_epoch(epoch)

    def records(self):
        with self._condition:
            return tuple(dict(record) for record in self._records)

    def wait_for_presentation_complete(self, epoch, deadline):
        """Block until the record for ``epoch`` exists or a finite deadline ends."""
        if type(epoch) is not int or isinstance(epoch, bool) or epoch < 1:
            raise ValueError('bounded turn epoch required')
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('finite wait deadline required')
        with self._condition:
            while True:
                self._expire()
                if self._stopped:
                    raise RuntimeError('presentation evidence unavailable: ' + str(self._reason))
                record = self._completed_epoch(epoch)
                if record is not None:
                    return dict(record)
                left = min(self.deadline, float(deadline)) - time.monotonic()
                if left <= 0:
                    raise TimeoutError('presentation completion deadline')
                self._condition.wait(min(.05, left))

    def stop(self, reason='stopped'):
        """Terminal: no further evidence is accepted and waiters are released."""
        if type(reason) is not str or not reason or len(reason) > 64:
            raise ValueError('bounded stop reason required')
        with self._condition:
            if self._stopped:
                return False
            self._stopped, self._reason = True, reason
            self._condition.notify_all()
            return True

    def snapshot(self):
        """Bounded JSON-ready diagnostics; no media, prompts or paths."""
        with self._condition:
            self._expire()
            return dict(schema=SNAPSHOT_SCHEMA, submitted_ends=tuple(self._ends),
                        completed_epochs=tuple(r['epoch'] for r in self._records),
                        playhead_seconds=self._playhead,
                        playhead_reports=self._playhead_reports,
                        rejections=self._rejections,
                        stopped=self._stopped, reason=self._reason,
                        deadline_left=max(0.0, self.deadline - time.monotonic()))
