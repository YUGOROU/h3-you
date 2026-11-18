"""Opt-in application endpoint + bounded speculative Pi; no GPU/device work on import.

The caller supplies an explicitly approved helper command and proposal callback.
Only the latest stopped application endpoint may select a proposal. It is NEVER
an Apple final. At most two speculative calls plus one endpoint call are allowed;
stale calls finish for budget reconciliation but never produce GPU requests.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import subprocess
import threading
import time

POLICY = 'application-stream-v1'
SOURCE = 'application_stable_partial'


class Endpoint:
    def __init__(self):
        self.revision = 0
        self.text = None
        self.endpoint = None

    def accept(self, row):
        kind = row.get('type')
        if kind not in ('partial', 'utterance_endpoint'):
            if 'error' in row:
                raise ValueError('streaming ASR failed; no endpoint committed')
            return None
        if self.endpoint is not None:
            raise ValueError('ASR text after committed endpoint')
        text, revision = row.get('text'), row.get('revision')
        if not isinstance(text, str) or not text.strip() or not 1 <= len(text) <= 160:
            raise ValueError('bounded nonempty streaming text required')
        if type(revision) is not int or revision < 1 or row.get('on_device') is not True or row.get('final') is not False:
            raise ValueError('explicit non-Apple-final on-device revision required')
        if kind == 'partial':
            if revision <= self.revision or row.get('capture_stopped') is not False:
                raise ValueError('nonmonotonic/live partial required')
            self.revision, self.text = revision, text
            return 'partial'
        if revision != self.revision or text != self.text:
            raise ValueError('endpoint differs from latest partial')
        if row.get('capture_stopped') is not True or row.get('endpoint_source') != SOURCE or row.get('audio_saved') is not False or row.get('cloud_fallback') is not False:
            raise ValueError('stopped private application endpoint required')
        self.endpoint = dict(row)
        return 'endpoint'


class StreamingASR:
    """Foreground owner. Context exit joins stale Pi calls for ledger reconciliation."""
    def __init__(self, propose, *, max_speculative=2):
        if type(max_speculative) is not int or not 0 <= max_speculative <= 2:
            raise ValueError('at most two speculative Pi calls')
        self.propose = propose
        self.limit = max_speculative
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='h3-pi-speculative')
        self.futures = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        # Never kill a running inference before its budget is reconciled.
        self.pool.shutdown(wait=True, cancel_futures=True)

    def process(self, events):
        """CPU-testable protocol; yielded events must come from a stopped-success helper."""
        endpoint = Endpoint()
        for row in events:
            event = endpoint.accept(row)
            if event == 'partial' and len(self.futures) < self.limit:
                key = (endpoint.revision, endpoint.text)
                self.futures[key] = self.pool.submit(self.propose, endpoint.text)
        if endpoint.endpoint is None:
            raise ValueError('no stopped application endpoint; no GPU request')
        key = (endpoint.revision, endpoint.text)
        selected = self.futures.get(key)
        proposal = selected.result(timeout=165) if selected else self.propose(endpoint.text)
        return endpoint.endpoint, proposal

    def capture(self, command, *, timeout=55):
        if self.closed or not 0 < timeout <= 55:
            raise ValueError('bounded open streaming owner required')
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        messages = queue.Queue(maxsize=1024)
        def read():
            try:
                for line in child.stdout:
                    if len(line) > 8192:
                        messages.put(ValueError('oversized ASR event'))
                        return
                    messages.put(line)
            finally:
                messages.put(None)
        reader = threading.Thread(target=read, name='h3-asr-events', daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        def events():
            count = 0
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('ASR capture deadline; no GPU request')
                try:
                    line = messages.get(timeout=left)
                except queue.Empty as exc:
                    raise TimeoutError('ASR capture deadline; no GPU request') from exc
                if line is None:
                    break
                if isinstance(line, Exception):
                    raise line
                count += 1
                if count > 1024:
                    raise ValueError('ASR event bound exceeded')
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('ASR event object required')
                yield row
            if child.wait(timeout=max(.1, deadline-time.monotonic())) != 0:
                raise ValueError('ASR helper failed; no GPU request')
        try:
            return self.process(events())
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
            # Drain bounded metadata without retaining or printing partial text.
            while reader.is_alive():
                try:
                    messages.get(timeout=.1)
                except queue.Empty:
                    pass
            reader.join()
            child.stdout.close()
