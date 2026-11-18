"""Offline by default; --run reads an already-enabled localhost OBS WebSocket.

Never enables the OBS server, changes settings/routing, starts camera/media,
captures PCM/images, joins Discord, allocates compute or reconnects. Optional
--meters observes only existing OBS volume metadata, not BlackHole/remote audio.
Password is read from an explicit environment variable, never argv or files.
"""
import argparse
import json
import os
import re

from research.h3_personal.obs_runtime import collect_audit
from research.h3_personal.obs_socket import LocalOBSWebSocket


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--meters', action='store_true')
    parser.add_argument('--port', type=int, default=4455)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--source-name', default='Verified silent generated video')
    parser.add_argument('--password-env', default='H3_OBS_PASSWORD')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.password_env):
        parser.error('environment variable name required, not a password')
    if not args.run:
        print(json.dumps(dict(mode='offline', network=False, read_only=True,
                              configuration_changes=False, gpu=False,
                              meter_subscription_requested=args.meters)))
        return 0
    try:
        transport = LocalOBSWebSocket(port=args.port, timeout=args.seconds)
    except (OSError, ValueError, EOFError):
        print(json.dumps(dict(read_only=True, runtime_verified=False,
                              remote_verified=False, errors=['localhost_connection_or_upgrade_failed'])))
        return 1
    report = collect_audit(transport, args.source_name,
                           password=os.environ.get(args.password_env),
                           meters=args.meters, timeout=args.seconds)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
