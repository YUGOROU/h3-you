#!/usr/bin/env python3
"""Stream read-only state from an existing Modal container, NOT discarded logs.

No GPU/model imports, environment dumps, tokens, media reads or Job mutation.
Remote observer is bounded even if the local connection disappears.
"""
import argparse
import json
import queue
import threading
import time
import re
import subprocess
import sys

REMOTE_PROBE = r'''
import datetime,json,math,time
from pathlib import Path

def document(name):
    path=Path(name)
    try:
        if path.is_symlink() or path.stat().st_size > 1024*1024:
            return {}
        value=json.loads(path.read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}

def number(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0

def label(value):
    import re
    return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value) else None

def inventory(root):
    count=0;size=0;latest=0;limited=False
    try:
        for path in Path(root).rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            count+=1
            if count>20000:
                limited=True;break
            stat=path.stat();size+=stat.st_size;latest=max(latest,stat.st_mtime)
    except OSError:
        limited=True
    return dict(files=min(count,20000),bytes=size,latest_write_unix=latest,scan_limited=limited)

def snapshot():
    boot=document('/tmp/h3-bootstrap-status.json')
    pipe=document('/work/identity/pipeline-report.json')
    train=document('/work/identity/training/training-snapshot.json')
    escrow=document('/work/identity/training/latest-escrow.json')
    worker=document('/work/identity/training/worker-status.json')
    request=document('/work/identity/train-request.json')
    result=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                bootstrap_stage=label(boot.get('stage')),pipeline_stage=label(pipe.get('stage')),
                failure_type=label(boot.get('failure_type') or pipe.get('failure_type')),
                runtime_ready=Path('/work/runtime-ready.json').is_file(),
                train_request_present=Path('/work/identity/train-request.json').is_file(),
                worker_status=label(worker.get('status')),
                cache=inventory('/work/identity/components'),model=inventory('/work/identity/model'))
    # No arbitrary nested receipt fields, paths or recorded content are emitted.
    for key,value in [('checkpoint_step',train.get('step')),('escrow_step',escrow.get('parent_step')),
                      ('max_updates',request.get('max_updates'))]:
        if number(value):result[key]=value
    for key in ('readiness','canonical_reload_verified','private_uploaded'):
        if type(pipe.get(key)) is bool:result[key]=pipe[key]
    return result

interval=float(__import__('sys').argv[1]);duration=float(__import__('sys').argv[2])
assert 5<=interval<=300 and 1<=duration<=3600
end=time.monotonic()+duration
while True:
    try:
        print(json.dumps(snapshot(),allow_nan=False),flush=True)
    except Exception as exc:
        print(json.dumps({'observer_error_type':type(exc).__name__}),flush=True)
    remaining=end-time.monotonic()
    if remaining<=0:break
    time.sleep(min(interval,remaining))
'''


def command(container, *, modal_cli, interval, seconds):
    if not re.fullmatch(r'ta-[A-Za-z0-9]+', container):
        raise ValueError('explicit existing ta- container ID required')
    if not 5 <= interval <= 300 or not 1 <= seconds <= 3600:
        raise ValueError('interval5–300s, total1–3600s required')
    return [modal_cli, 'container', 'exec', '--no-pty', '--', container,
            '/opt/conda/bin/python', '-u', '-c', REMOTE_PROBE, str(interval), str(seconds)]


class ProgressDisplay:
    def __init__(self, factory):
        self.bars = {key: factory(desc=title, total=None, unit=unit, unit_scale=scale,
                                 position=i, dynamic_ncols=True, leave=True)
                     for i, (key, title, unit, scale) in enumerate([
                         ('cache', 'Cache size', 'B', True), ('model', 'Model size', 'B', True),
                         ('checkpoint_step', 'Saved step (cap)', 'step', False),
                         ('escrow_step', 'Escrow step', 'step', False)])}

    def update(self, value):
        stage = value.get('pipeline_stage') or value.get('bootstrap_stage') or 'unknown'
        for key, bar in self.bars.items():
            entry = value.get(key)
            count = entry.get('bytes') if isinstance(entry, dict) else entry
            if type(count) not in (int, float) or count < 0:
                continue
            if key == 'checkpoint_step' and type(value.get('max_updates')) is int:
                bar.total = value['max_updates']  # ceiling, NOT expected convergence/completion
            # Inventory can shrink on cleanup; it is not network-transfer throughput.
            bar.n = count
            extra = {'stage': stage}
            if isinstance(entry, dict):
                extra.update(files=entry.get('files'), limited=entry.get('scan_limited'))
            bar.set_postfix(extra, refresh=False)
            bar.refresh()
        if value.get('failure_type') or value.get('observer_error_type'):
            self.bars['cache'].write('Error: ' + str(value.get('failure_type') or value.get('observer_error_type')))

    def close(self):
        for bar in self.bars.values():
            bar.close()


def render_stream(child, display, timeout):
    messages = queue.Queue(maxsize=8)
    def read_lines():
        for line in iter(lambda: child.stdout.readline(65537), ''):
            if len(line) <= 65536:
                messages.put(line)
        messages.put(None)
    threading.Thread(target=read_lines, daemon=True).start()
    end = time.monotonic() + timeout
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(child.args, timeout)
        try:
            line = messages.get(timeout=min(1, remaining))
        except queue.Empty:
            continue
        if line is None:
            return child.wait(timeout=max(.1, end-time.monotonic()))
        try:
            value = json.loads(line)
        except ValueError:
            continue  # CLI notices are not state snapshots; stderr remains visible.
        if isinstance(value, dict):
            display.update(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('container', help='existing ta- container ID (not App/Call ID)')
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--seconds', type=int, default=600, help='finite observation duration, max3600')
    parser.add_argument('--modal-cli', default='modal')
    parser.add_argument('--json', action='store_true', help='raw JSON stream; no tqdm dependency')
    args = parser.parse_args()
    try:
        cmd = command(args.container, modal_cli=args.modal_cli, interval=args.interval, seconds=args.seconds)
    except ValueError as exc:
        parser.error(str(exc))
    print('Read-only state stream; discarded stdout/stderr and per-update loss are NOT recoverable.',
          file=sys.stderr)
    display = None
    if not args.json:
        try:
            from tqdm import tqdm
        except ImportError:
            parser.error('use uv run --with tqdm, or --json for dependency-free output')
        display = ProgressDisplay(tqdm)
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE if display else None,
                             text=True, bufsize=1)
    try:
        return (render_stream(child, display, args.seconds + 90) if display else
                child.wait(timeout=args.seconds + 90))
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        # Stop the observer CLI, NEVER the App/container/trainer. Remote loop has its own deadline.
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill();child.wait()
        print('Observer stopped; Job was not stopped. Remote observer ends at its deadline.',file=sys.stderr)
        return 130
    finally:
        if display:
            display.close()


if __name__ == '__main__':
    sys.exit(main())
