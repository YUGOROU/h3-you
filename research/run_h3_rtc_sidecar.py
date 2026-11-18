"""Private FD/config entrypoint; never print credentials, SDP or exception bodies."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    from research.h3_personal.webrtc_sidecar import serve
    from research.h3_personal.paired_av_wire import _pairs
    os.umask(0o077)
    if len(sys.argv) != 3:
        raise ValueError('metadata path and owned FD required')
    fd = int(sys.argv[2])
    if fd < 0:
        raise ValueError('owned FD required')
    config = json.loads(Path(sys.argv[1]).read_text(), object_pairs_hook=_pairs)
    if set(config) != {'binding', 'expires_at', 'deadline', 'port'}:
        raise ValueError('exact sidecar metadata required')
    remaining = min(config['deadline'] - time.monotonic(), config['expires_at'] - time.time())
    if not 0 < remaining <= 1200:
        raise ValueError('finite sidecar lease required')
    # Hard expiry also bounds synchronous library cleanup / blocked executor calls.
    signal.setitimer(signal.ITIMER_REAL, remaining)
    asyncio.run(serve(config, fd))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        raise SystemExit(1) from None
