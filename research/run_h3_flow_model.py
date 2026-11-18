"""Offline conditional pipeline model from saved evidence; no media or network run."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

from .h3_personal.av_flow_model import FlowCase, FlowModelError, simulate
from .h3_personal.transport_replay_preflight import _read, _unique
from .h3_personal.webrtc_saved_source import SavedSource


def _json(path):
    return json.loads(_read(Path(path), 262144), object_pairs_hook=_unique)


def _time(row, key):
    x = row.get(key)
    if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1e12:
        raise FlowModelError('finite local evidence time required')
    return x


def _rows(value, name, count):
    if type(value) is not list or len(value) != count:
        raise FlowModelError('matching ordered evidence required')
    for i, row in enumerate(value):
        if type(row) is not dict or type(row.get(name)) is not int or row[name] != i:
            raise FlowModelError('exact evidence indices required')
    return value


def model_evidence(capture_dir, producer_path, receiver_path, sink_path):
    # This validates original bytes/index sizes, not a cryptographic association
    # of legacy independent timing receipts. No optional media imports occur.
    source = SavedSource(capture_dir, timing_path=producer_path)
    count = len(source.envelopes)
    producer = _rows(_json(producer_path)['windows'], 'index', count)
    receiver_value = _json(receiver_path)
    if receiver_value.get('clock') != 'local_monotonic':
        raise FlowModelError('receiver-local evidence required')
    receiver = _rows(receiver_value.get('receive'), 'index', count)
    events = _rows(_json(sink_path).get('events'), 'sequence', count)
    pulls, media, service, decode, rtts, observed_waits = [], [], [], [], [], []
    for i, (p, r, e) in enumerate(zip(producer, receiver, events)):
        pkeys = ('credit_wait_started', 'pull_started', 'packet_ready', 'ack_verified', 'ack_sink_accepted')
        pv = [_time(p, k) for k in pkeys]
        rv = [_time(r, k) for k in ('received', 'verified', 'accept_start', 'accepted', 'sink_acked')]
        if pv != sorted(pv) or rv != sorted(rv):
            raise FlowModelError('nonmonotonic local evidence')
        envelope = source.envelopes[i]
        if any(type(e.get(k)) is not int or e[k] != getattr(envelope, k)
               for k in ('frame_start', 'frame_stop', 'sample_start', 'sample_stop')):
            raise FlowModelError('sink media indices mismatch')
        duration = (envelope.sample_stop - envelope.sample_start) / 48000
        if abs(_time(e, 'media_seconds') - duration) > 1e-6:
            raise FlowModelError('sink media duration mismatch')
        pulls.append(pv[2] - pv[1])
        observed_waits.append(pv[1] - pv[0])
        rtts.append(pv[3] - pv[2])
        media.append(duration)
        service.append(rv[3] - rv[2])
        decode.append(_time(e, 'decode_seconds'))
    final_prepare = _time(receiver[-1], 'sink_acked') - _time(receiver[-1], 'accepted')
    base = FlowCase(tuple(pulls), tuple(media), tuple(service), tuple(decode), finalization=final_prepare)
    half_proxy = statistics.mean(rtts) / 2
    cases = (
        ('zero-carrier-delay-bound', base),
        ('assumed-20ms-roundtrip-NOT-measured-WebRTC', replace(base, forward_delay=.01, reverse_delay=.01)),
        ('observed-verification-RTT-mean-half-split-PROXY',
         replace(base, forward_delay=half_proxy, reverse_delay=half_proxy)),
    )
    output = dict(schema='h3-av-flow-analysis-v1', network_executed=False,
                  gpu_executed=False, playback_verified=False,
                  evidence_binding='operator-selected/index-size-matched; not cryptographically bound',
                  parameter_caveat='fixed observed pull/sink durations; assumed transport splits; finalization uses receiver accepted-to-ACK-enqueued proxy, not isolated finish; not causal forecasts',
                  observed_final_accept_to_ack_enqueued_seconds=final_prepare,
                  pulls_seconds=pulls, historical_credit_wait_seconds=observed_waits,
                  measured_verification_roundtrip_seconds=rtts,
                  sink_service_seconds=service, sink_decode_seconds=decode,
                  source_manifest_sha256=source.preflight['source_manifest_sha256'],
                  cases={name: simulate(case) for name, case in cases})
    if len(pulls) > 4:
        steady = statistics.median(pulls[4:])
        output['counterfactual_steady_median_pull_seconds'] = steady
        output['cases']['steady-median-pulls-zero-delay-COUNTERFACTUAL'] = simulate(
            replace(base, pulls=(steady,) * count))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', required=True)
    parser.add_argument('--producer-timing', required=True)
    parser.add_argument('--receiver-timing', required=True)
    parser.add_argument('--sink-timing', required=True)
    args = parser.parse_args(argv)
    try:
        result = model_evidence(args.capture_dir, args.producer_timing, args.receiver_timing, args.sink_timing)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, RecursionError):
        print(json.dumps(dict(schema='h3-av-flow-error-v1', error='invalid saved evidence; no execution or retry')))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
