"""Bounded in-process worker recovery; no provider, device or launch authority."""
import time


def plan_turn_recovery(sequence, fault, action, *, state, endpoint, delivery,
                       complete, render, playback):
    """Validate recovery before scheduling; never re-send an uncertain request."""
    phase = fault['phase']
    video_sequence = state['turns'][str(sequence)].get('video_sequence')
    if action == 'skip':
        safe_capture = phase == 'capture' and endpoint and not endpoint.closed
        played = delivery.turns.get(video_sequence, {}) if delivery else {}
        safe_play = phase == 'playback' and 'play_end' in played
        if not (safe_capture or safe_play):
            raise ValueError('cannot safely skip uncertain request')
        ready_status = 'ready' if delivery else 'asr_ready_preflight'
        return lambda: complete(sequence, 'skipped_failure', ready_status)
    if action != 'retry':
        raise ValueError('retry or skip required')
    if phase in ('generation_submit', 'generation_finish') and delivery:
        pending = delivery.recover_pending(video_sequence)
        if time.monotonic() >= pending['end']:
            raise ValueError('original generation wait expired')
        return lambda: render(None, None, sequence, video_sequence, pending)
    if phase == 'playback' and delivery:
        asset = delivery.recover_asset(video_sequence)
        return lambda: playback(asset, sequence, video_sequence)
    raise ValueError('requires operator diagnosis; no safe same-session retry')


class ResidentRecovery:
    def __init__(self, state, lock, persist, *, maximum_attempts=2):
        self.state, self.lock, self.persist = state, lock, persist
        self.maximum_attempts = maximum_attempts
        self.workers = []
        state.update(failures={}, recovery_attempts=0)

    def submit(self, pool, function, *args):
        # Registration and recovery admission share a lock, including nested playback.
        with self.lock:
            future = pool.submit(function, *args)
            self.workers = [f for f in self.workers if not f.done()]
            self.workers.append(future)
            return future

    def fail(self, sequence, phase, exc):
        with self.lock:
            self.state['failures'][str(sequence)] = dict(phase=phase, error_type=type(exc).__name__)
            self.state['turns'][str(sequence)]['stage'] = 'failed'
            self.state.update(status='degraded', requires_operator_review=True,
                              error_type=type(exc).__name__)
            self.persist()

    def complete(self, sequence, stage, ready_status='ready'):
        with self.lock:
            self.state['turns'][str(sequence)]['stage'] = stage
            self.state['failures'].pop(str(sequence), None)
            if not self.state['failures']:
                self.state.update(status=ready_status, requires_operator_review=False)
                self.state.pop('error_type', None)
            else:
                self.state.update(status='degraded', requires_operator_review=True)
            self.persist()

    def recover(self, sequence, prepare, pool, *, now, deadline, stopping=False):
        """prepare validates an exact existing attempt and returns a no-launch task.

        No retry is automatic. All workers must be quiescent, and neither the
        session deadline nor any per-generation deadline is extended.
        """
        with self.lock:
            if type(sequence) is not int or str(sequence) not in self.state['failures']:
                raise ValueError('known failed turn required')
            if stopping or now >= deadline:
                raise ValueError('session stopping or expired')
            if any(not f.done() for f in self.workers):
                raise ValueError('workers still active')
            if self.state['recovery_attempts'] >= self.maximum_attempts:
                raise ValueError('bounded session recovery limit')
            fault = dict(self.state['failures'][str(sequence)])
            task = prepare(fault)  # Must not submit, replay, capture, or launch.
            self.state['recovery_attempts'] += 1
            self.state.update(status='recovering', requires_operator_review=False)
            self.persist()

            def run():
                try:
                    task()
                except BaseException as exc:
                    self.fail(sequence, fault['phase'], exc)
            try:
                return self.submit(pool, run)
            except BaseException as exc:
                self.fail(sequence, fault['phase'], exc)
                raise
