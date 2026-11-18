"""CPU-only synthetic browser preview. Offline unless --serve is explicit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.h3_personal.browser_av import BrowserAVServer, hls_arguments
from research.h3_personal.core_audio_output import output_status

LABEL = 'synthetic_noH3'


def encoder_arguments(media_root):
    return ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-re', '-f', 'lavfi', '-i', 'testsrc2=size=252x448:rate=24',
            '-re', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono',
            '-t', '8'] + hls_arguments(media_root)


def _private_root(directory):
    root = Path(directory).expanduser().absolute()
    repo = Path(__file__).resolve().parents[1]
    if root.resolve().is_relative_to(repo):
        raise ValueError('preview directory must be outside repository')
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('symlink paths forbidden')
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(root, 0o700)
    return root


def serve(directory, *, status=output_status, server_factory=BrowserAVServer,
          popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep):
    """Finite explicit preview; injectable device/server/process seams for tests."""
    root = _private_root(directory)
    result = dict(label=LABEL, outcome='failed', gpu=False)
    process = None
    try:
        if status().get('default_is_builtin_speaker') is not True:
            raise RuntimeError('Default output is not a verified built-in speaker; no routing changed.')
        # startup_seconds is the server's separate first-Connect lease, not renewal.
        with server_factory(root, status, max_seconds=240, startup_seconds=120) as server:
            absolute = clock() + 360
            print(f'{LABEL}: {server.url}', flush=True)
            print('Open URL manually. Connect/Prepare, then Play. Silent synthetic AV; no H3/GPU.', flush=True)
            connect_until = min(absolute, clock() + 120)
            while not server.client_ready.is_set():
                if server.stream_state == 'failed' or clock() >= connect_until:
                    raise RuntimeError('Connect deadline or cancellation')
                sleep(.05)
            server.validate_output()
            server.media_root.mkdir(mode=0o700)
            server.finish('streaming')
            active_until = min(absolute, clock() + 240)
            with open(root / 'encoder-stderr.txt', 'xb') as errors:
                os.chmod(root / 'encoder-stderr.txt', 0o600)
                process = popen(encoder_arguments(server.media_root), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=errors)
                try:
                    while process.poll() is None:
                        if not server.playback_done.is_set():
                            server.validate_output()
                        if clock() >= active_until:
                            raise RuntimeError('encoder deadline')
                        sleep(.05)
                    if process.returncode != 0:
                        raise RuntimeError('synthetic encoder failed')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1)
                    else:
                        process.wait(timeout=1)
            playlist = server.media_root / 'index.m3u8'
            if b'#EXT-X-ENDLIST' not in playlist.read_bytes():
                raise RuntimeError('missing real encoder ENDLIST')
            server.finish('eos')
            while not server.playback_done.is_set():
                if server.stream_state == 'failed' or clock() >= active_until:
                    raise RuntimeError('playback deadline or cancellation')
                sleep(.05)
            result['outcome'] = 'ended'
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            else:
                process.wait(timeout=1)
        fd = os.open(root / 'preview-result.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as out:
            json.dump(result, out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--directory')
    args = parser.parse_args(argv)
    if not args.serve:
        print(json.dumps(dict(label=LABEL, mode='offline_plan', gpu=False,
                              next='Explicit --serve --directory NEW_PRIVATE_PATH; manual Connect/Prepare then Play')))
        return 0
    if not args.directory:
        parser.error('--serve requires --directory (fresh, outside repository)')
    try:
        serve(args.directory)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Preview stopped: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
