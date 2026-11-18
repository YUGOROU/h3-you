"""Finite two-drain HTTPS Vox restoration on the existing L4 Endpoint.

Offline by default. No H3/ASR/device/browser/Discord or saved media output. Old
speech is drained and quarantined, not prematurely closed; second request reuses
that same healthy native service. This is not fresh conversation acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from research.h3_personal.voxcpm_https_lease import lease, PRIVATE
from research.h3_personal.voxcpm_endpoint_client import RemoteVoxCPM
from research.run_h3_irodori_endpoint import persist

TEXTS = ('おはようございます。今日は落ち着いて始めましょう。',
         'わかりました。新しいお話を聞かせてください。')


def drain_two(source, *, client_factory=RemoteVoxCPM):
    config = source['config']
    client = client_factory(config['url'], token=source['token'], key=source['key'],
        session=config['session'], boot=config['boot'], expires_at=config['expires_at'])
    rows = []
    try:
        client.read('tts_ready', time.monotonic()+30)
        for epoch, text in enumerate(TEXTS, 1):
            started = time.monotonic(); first = None; digest = hashlib.sha256()
            samples = chunks = 0
            deadline = min(started+120, started+config['expires_at']-time.time()-15)
            for pcm in client.stream(text, epoch, deadline):
                if first is None:first = time.monotonic()
                digest.update(pcm); samples += len(pcm)//4; chunks += 1
            if first is None:
                raise ValueError('native speech PCM required')
            rows.append(dict(epoch=epoch, samples=samples, chunks=chunks,
                sha256=digest.hexdigest(), first_pcm_seconds=first-started,
                total_seconds=time.monotonic()-started,
                disposition='quarantined-old-epoch' if epoch == 1 else 'verified-new-source',
                protocol_END_verified=True, playback_verified=False))
        evidence = client._request('GET', '/evidence', None, time.monotonic()+15)
        completions = evidence['evidence']['completions']
        if len(completions) != 2:
            raise ValueError('exact serial two-request native evidence required')
        for row, native in zip(rows, completions):
            if (native['epoch'] != row['epoch'] or native['samples'] != row['samples']
                    or native['chunks'] != row['chunks']
                    or native['native_float_sha256'] != row['sha256']):
                raise ValueError('native-to-HTTPS exact PCM evidence mismatch')
            completion = native['completion']
            row['native_natural_stop_verified'] = (
                completion.get('reason') == 'natural_stop'
                and completion.get('natural_stop') is True
                and completion.get('stream_exhausted') is True)
        return dict(streams=rows, evidence=evidence, source_complete=True,
                    exact_native_pcm_verified=True, serial_native_service_reused=True,
                    transport='https-native-v1', media_saved=False,
                    fresh_conversation=False, playback_verified=False)
    finally:
        client.close()


def run(directory, *, parent_campaign=None):
    root = Path(os.path.abspath(directory))
    if (root.exists() or not root.is_relative_to(PRIVATE)
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError('fresh private campaign required')
    os.umask(0o077); root.mkdir(mode=0o700)
    result = dict(schema='h3-https-source-restoration-probe-v1', complete=False,
                  devices_used=False, discord_used=False, cleanup_confirmed=False)
    try:
        with lease(root/'lease', parent_campaign=parent_campaign) as source:
            result.update(drain_two(source))
        result['complete'] = True
    except BaseException as error:
        result['error_type'] = type(error).__name__
        raise
    finally:
        terminal = root/'lease/terminal.json'
        if terminal.is_file():
            row = json.loads(terminal.read_text())
            result['cleanup_confirmed'] = (row.get('state') == 'paused'
                                          and row.get('independent_confirmation') is True)
        persist(root/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--parent-campaign', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(scope='offline-only', endpoint='existing-Vox-L4',
                              max_seconds=1200, max_requests=2, devices=False)))
        return 0
    if args.directory is None:parser.error('--directory required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('terminated')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('finite owner deadline')))
    signal.setitimer(signal.ITIMER_REAL, 1500)
    try:
        result = run(args.directory, parent_campaign=args.parent_campaign)
        print(json.dumps(dict(type='https_source_terminal', complete=result['complete'],
                              cleanup_confirmed=result['cleanup_confirmed'])), flush=True)
        return 0
    except BaseException as error:
        print(json.dumps(dict(type='https_source_failed', error_type=type(error).__name__)), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    raise SystemExit(main())
