"""Default offline; --run starts one hard-bounded synthetic WebRTC child."""
import argparse
import asyncio
import json
import logging
import subprocess
import sys

from .h3_personal.transport_replay_preflight import capabilities
from .h3_personal.webrtc_synthetic_probe import run_synthetic, validate_duration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--duration', type=float, default=4)
    parser.add_argument('--capture-dir', help='private saved source, same-host only')
    parser.add_argument('--producer-timing', help='private same-origin packet_ready schedule')
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        duration = validate_duration(args.duration)
        if args.producer_timing and not args.capture_dir:
            raise ValueError('timing requires saved source')
        if args.capture_dir and duration != 4:
            raise ValueError('saved source cannot be clipped with duration')
        if not args.run:
            if args._child:
                raise ValueError('child requires explicit run')
            print(json.dumps(dict(schema='h3-webrtc-probe-offline-v1', dependencies=capabilities(),
                                  transport_executed=False, dependencies_installed=False)))
            return 0
        if args._child:
            # Never emit library SDP/ICE/address/credential diagnostics.
            for name in ('aiortc', 'aioice'):
                logging.getLogger(name).setLevel(logging.CRITICAL)
            if args.capture_dir:
                from .h3_personal.webrtc_saved_source import run_saved
                result = asyncio.run(run_saved(args.capture_dir, timing_path=args.producer_timing))
            else:
                result = asyncio.run(run_synthetic(duration_seconds=duration))
            print(json.dumps(result, sort_keys=True))
            return 0
        command = [sys.executable, '-m', 'research.run_h3_webrtc_probe',
                   '--run', '--_child', '--duration', str(duration)]
        if args.capture_dir:
            command.extend(['--capture-dir', args.capture_dir])
        if args.producer_timing:
            command.extend(['--producer-timing', args.producer_timing])
        hard_timeout = 100 if args.capture_dir else 30
        child = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               timeout=hard_timeout, check=False)
        if child.returncode != 0 or len(child.stdout) > 65536:
            raise ValueError('probe child failed')
        value = json.loads(child.stdout)
        expected_schema = 'h3-webrtc-saved-probe-v1' if args.capture_dir else 'h3-webrtc-synthetic-probe-v1'
        if (type(value) is not dict or value.get('schema') != expected_schema
                or value.get('cleanup_complete') is not True):
            raise ValueError('invalid probe receipt')
        value['parent_hard_timeout_seconds'] = hard_timeout
        print(json.dumps(value, sort_keys=True))
        return 0
    except Exception:
        # No raw library errors, local candidates, SDP, paths or credentials.
        print(json.dumps(dict(schema='h3-webrtc-probe-error-v1',
                              error='finite synthetic probe failed; no fallback or retry')))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
