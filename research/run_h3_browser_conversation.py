"""Fresh typed diagnostic browser owner. Offline plan unless explicitly --run.

No ASR, capture, OBS, routing changes, old owner replay, or automatic browser launch.
Use the existing HF Python environment for an explicitly approved live operation.
"""
import argparse
import json
import os
from pathlib import Path
import time

READINESS_SECONDS = 600


def plan():
    return dict(schema='h3-browser-owner-v1', input='two_fresh_operator_diagnostic_texts_not_asr',
                reference='original', tts_mode='complete', anchor='every3',
                startup_seconds=1200, active_seconds=240, readiness_seconds=READINESS_SECONDS,
                capture=False, obs=False, routing_changes=False)


def own(directory, turns, *, server_factory=None, run=None, status=None,
        sink_factory=None, clock=time.monotonic, sleep=time.sleep, emit=print):
    """Own one fresh server through run's sink natural EOS and cleanup.

    The direct launcher requires a nonexistent directory (older variants permit
    exist_ok); a fresh child avoids depending on either version's mkdir policy.
    The sink keeps the launcher's receiver root for pump ownership; HTTP media
    remains in the enclosing server's private root.
    """
    if (not isinstance(turns, (list, tuple)) or len(turns) != 2
            or any(not isinstance(t, str) or not t.strip() or len(t) > 2000 for t in turns)):
        raise ValueError('two fresh bounded operator diagnostic texts required')
    root = Path(directory).absolute()
    repo = Path(__file__).resolve().parents[1]
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('nonsymlink private root required')
    if root.resolve().is_relative_to(repo) or root.exists():
        raise ValueError('fresh private root outside repository required; never replay')
    root.mkdir(mode=0o700, parents=True)
    os.chmod(root, 0o700)
    outcome = 'failed'
    try:
        # Imports which touch HF/runtime dependencies occur only on explicit run.
        if server_factory is None or sink_factory is None:
            from research.h3_personal.browser_av import BrowserAVServer, BrowserAVSink
            server_factory = server_factory or BrowserAVServer
            sink_factory = sink_factory or BrowserAVSink
        if status is None:
            from research.h3_personal.core_audio_output import output_status
            status = output_status
        with server_factory(root, status, startup_seconds=1200, max_seconds=240) as server:
            emit('H3 browser diagnostic: ' + server.url)
            emit('Open manually; Prepare then Play. Two typed diagnostic turns, not ASR.')
            deadline = clock() + READINESS_SECONDS
            while not server.client_ready.is_set():
                if server.stream_state in ('failed', 'ended'):
                    raise RuntimeError('browser cancelled or failed before readiness')
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError('browser readiness deadline')
                sleep(min(.05, remaining))
            server.validate_output()
            if run is None:
                from research.launch_h3_direct_conversation import run

            def local_sink(receiver_root, binding, **kwargs):
                server.validate_output()
                return sink_factory(receiver_root, binding, server=server,
                                    playback_approved=True, **kwargs)

            result = run(root / 'conversation', list(turns),
                         local_sink_factory=local_sink, idle_anchor=False,
                         conversation_anchor=True, tts_mode='complete',
                         capture_comet=False, asr_parent=None, allow_obs=False,
                         obs_port=None, direct_audio_binary=None)
            if not server.playback_done.is_set() or server.stream_state != 'ended':
                raise RuntimeError('browser natural ended required')
            outcome = 'ended'
            return result
    finally:
        # Explicit allowlist: never serialize turns, URL, token, exception or run result.
        receipt = dict(plan(), outcome=outcome)
        fd = os.open(root / 'browser-owner.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                     0o600)
        with os.fdopen(fd, 'w') as out:
            json.dump(receipt, out, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(plan(), sort_keys=True))
        return
    if args.directory is None:
        parser.error('--run requires --directory (fresh private root)')
    turns = [input('Fresh operator diagnostic text %d (not ASR): ' % i) for i in (1, 2)]
    own(args.directory, turns)


if __name__ == '__main__':
    main()
