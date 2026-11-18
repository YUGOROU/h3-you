"""Deterministic conditional CPU model, NOT GPU/network/playback evidence.

One serial generator, FIFO carrier/ACK propagation, one serial sink, two slots.
Generation occupies a slot. Only sink-ACK receipt releases it. Historical ready
or credit-wait times are never replayed: the event loop recomputes both.
"""
from dataclasses import dataclass
import heapq
import math


class FlowModelError(ValueError):
    pass


def _number(value, upper=1200):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
        raise FlowModelError('finite nonnegative bounded model duration required')
    return float(value)


@dataclass(frozen=True)
class FlowCase:
    pulls: tuple
    media: tuple
    sink_service: tuple
    decode: tuple
    forward_delay: float = 0.0
    reverse_delay: float = 0.0
    finalization: float = 0.0
    deadline: float = 1200.0
    withhold_final_ack: bool = False

    def __post_init__(self):
        n = len(self.pulls) if type(self.pulls) is tuple else 0
        if not 1 <= n <= 144:
            raise FlowModelError('bounded nonempty pull tuple required')
        for vector in (self.pulls, self.media, self.sink_service, self.decode):
            if type(vector) is not tuple or len(vector) != n:
                raise FlowModelError('equal bounded vectors required')
            for x in vector: _number(x)
        if any(x <= 0 for x in self.media) or any(d > s for d, s in zip(self.decode, self.sink_service)):
            raise FlowModelError('positive media and decode within sink service required')
        for x in (self.forward_delay, self.reverse_delay, self.finalization, self.deadline): _number(x)
        if self.deadline == 0 or type(self.withhold_final_ack) is not bool:
            raise FlowModelError('finite deadline and explicit final ACK policy required')


def simulate(case):
    if type(case) is not FlowCase:
        raise FlowModelError('explicit FlowCase required')
    case.__post_init__()
    n = len(case.pulls)
    rows = [dict(index=i) for i in range(n)]
    heap = []
    serial = scheduled = processed = 0
    pending = []
    waiting = []
    generating = active_sink = None
    next_pull = next_sink = 0
    wanted = 0.0
    now = forward_cursor = reverse_cursor = 0.0
    high_slots = high_queue = 0
    finalized = complete = False
    terminal = None

    def add(at, event, index=-1):
        nonlocal serial, scheduled
        if not math.isfinite(at) or at < now or scheduled >= 4096:
            raise FlowModelError('invalid or excessive model events')
        heapq.heappush(heap, (at, serial, event, index))
        serial += 1
        scheduled += 1

    def check_slots():
        nonlocal high_slots, high_queue
        slots = len(pending) + int(generating is not None)
        if slots > 2 or len(waiting) > 1:
            raise FlowModelError('two-slot or one-prefetch bound violated')
        high_slots = max(high_slots, slots)
        high_queue = max(high_queue, len(waiting))

    def maybe_pull():
        nonlocal next_pull, generating
        if next_pull < n and generating is None and len(pending) < 2:
            i = next_pull
            rows[i].update(credit_wait_started=wanted, pull_started=now,
                           credit_wait_seconds=now-wanted)
            generating = i
            next_pull += 1
            add(now + case.pulls[i], 'ready', i)
            check_slots()

    def ack(event, i):
        nonlocal reverse_cursor
        at = max(now + case.reverse_delay, reverse_cursor)
        reverse_cursor = at
        rows[i][event + '_sent'] = now
        add(at, event, i)

    def maybe_sink():
        nonlocal active_sink, next_sink
        if active_sink is None and waiting:
            i = waiting.pop(0)
            if i != next_sink:
                raise FlowModelError('sink order violated')
            next_sink += 1
            active_sink = i
            rows[i]['accept_started'] = now
            add(now + case.decode[i], 'decoded', i)
            add(now + case.sink_service[i], 'accepted', i)
            check_slots()

    add(case.deadline, 'deadline')
    maybe_pull()
    while heap:
        now, _, event, i = heapq.heappop(heap)
        processed += 1
        if now >= case.deadline:
            terminal = 'deadline; no replay'
            break
        if event == 'ready':
            if generating != i:
                raise FlowModelError('multiple generator owners')
            generating = None
            pending.append(i)
            rows[i]['packet_ready'] = now
            at = max(now + case.forward_delay, forward_cursor)
            forward_cursor = at
            add(at, 'verified', i)
            wanted = now
            maybe_pull()
        elif event == 'verified':
            rows[i]['received_verified'] = now
            ack('verified_ack', i)  # Deliberately no credit release.
            waiting.append(i)
            maybe_sink()
        elif event == 'verified_ack':
            rows[i]['verified_ack_received'] = now
        elif event == 'decoded':
            rows[i]['decoded_ready'] = now
        elif event == 'accepted':
            if active_sink != i:
                raise FlowModelError('sink lease mismatch')
            rows[i]['accepted'] = now
            active_sink = None
            if i == n-1:
                add(now + case.finalization, 'finalized', i)
            else:
                ack('sink_ack', i)
            maybe_sink()
        elif event == 'finalized':
            finalized = True
            rows[i]['finalized'] = now
            if not case.withhold_final_ack:
                ack('sink_ack', i)
        elif event == 'sink_ack':
            if not pending or pending[0] != i or 'verified_ack_received' not in rows[i]:
                raise FlowModelError('FIFO verified/sink ACK contract violated')
            pending.pop(0)
            rows[i]['sink_ack_received'] = now
            maybe_pull()
            if finalized and next_pull == n and not pending and generating is None:
                complete = True
                break
        else:
            raise FlowModelError('unknown model event')
        check_slots()

    unfinished_credit_wait = 0.0
    if terminal is not None and generating is None and next_pull < n:
        unfinished_credit_wait = max(0.0, now - wanted)
        rows[next_pull].update(credit_wait_started=wanted,
                               credit_wait_incomplete_seconds=unfinished_credit_wait)

    # A hypothetical window player: consumes each complete decoded window at its
    # media rate, stalls if unavailable; separate from pipe acceptance/ACK.
    first = next((r['decoded_ready'] for r in rows if 'decoded_ready' in r), None)
    virtual_end = first
    nominal_elapsed = cumulative_idle = peak_deficit = sink_idle = 0.0
    last_accepted = None
    for i, row in enumerate(rows):
        # Sink acceptance and decode readiness are separate event evidence.
        if 'accept_started' in row and last_accepted is not None:
            sink_idle += max(0.0, row['accept_started'] - last_accepted)
        if 'accepted' in row:
            last_accepted = row['accepted']
        if 'decoded_ready' not in row:
            continue
        deficit = max(0.0, row['decoded_ready'] - first - nominal_elapsed)
        idle = max(0.0, row['decoded_ready'] - virtual_end)
        row.update(virtual_window_idle_seconds=idle, fixed_deadline_deficit_seconds=deficit)
        cumulative_idle += idle
        peak_deficit = max(peak_deficit, deficit)
        virtual_end = max(virtual_end, row['decoded_ready']) + case.media[i]
        nominal_elapsed += case.media[i]
    unfinished_sink_idle = (max(0.0, now - last_accepted)
                            if terminal is not None and active_sink is None
                            and next_sink < n and last_accepted is not None else 0.0)
    unfinished_playout_idle = 0.0
    if terminal is not None and virtual_end is not None and sum('decoded_ready' in r for r in rows) < n:
        unfinished_playout_idle = max(0.0, now - virtual_end)
        peak_deficit = max(peak_deficit, max(0.0, now - first - nominal_elapsed))
    completed_credit_wait = sum(r.get('credit_wait_seconds', 0) for r in rows)
    return dict(schema='h3-av-flow-model-v1', evidence='conditional discrete-event model only',
                complete=complete, terminal=terminal, virtual_seconds=now,
                final_sink_finalized=finalized, pending=len(pending),
                generating=generating is not None, slots_highwater=high_slots,
                prefetch_highwater=high_queue, events_processed=processed,
                forward_delay_assumption=case.forward_delay,
                reverse_delay_assumption=case.reverse_delay,
                credit_wait_seconds=completed_credit_wait + unfinished_credit_wait,
                completed_credit_wait_seconds=completed_credit_wait,
                incomplete_credit_wait_seconds=unfinished_credit_wait,
                sink_idle_seconds=sink_idle + unfinished_sink_idle,
                incomplete_sink_idle_seconds=unfinished_sink_idle,
                virtual_window_playout_idle_seconds=cumulative_idle + unfinished_playout_idle,
                incomplete_virtual_playout_idle_seconds=unfinished_playout_idle,
                max_fixed_media_deadline_deficit_seconds=peak_deficit,
                deadline_deficit_is_lower_bound=terminal is not None,
                virtual_player_scope='hypothetical decoded-window consumption, not presentation or memory accounting',
                playback_verified=False, network_measured=False, gpu_executed=False,
                rows=rows)
