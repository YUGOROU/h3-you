"""Offline only: inspect replay inputs and optional dependency availability."""
import argparse
import json

from .h3_personal.transport_replay_preflight import (
    ReplayPreflightError, capabilities, inspect_capture,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', help='private directory of saved 000.json/mp4/pcm triples')
    args = parser.parse_args(argv)
    result = dict(schema='h3-transport-capabilities-v1', dependencies=capabilities(),
                  network_executed=False, device_access=False, dependencies_installed=False)
    try:
        if args.capture_dir:
            result['capture'] = inspect_capture(args.capture_dir)
    except ReplayPreflightError:
        print(json.dumps(dict(schema='h3-transport-preflight-error-v1',
                              error='invalid capture; no transport executed')))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
