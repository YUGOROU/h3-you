"""Loopback-only persistent keyboard image review. No cloud/job APIs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
from urllib.parse import urlparse
from .preprocess import dump

PAGE = Path(__file__).with_name('review_page.html').read_text()


class ReviewStore:
    def __init__(self, directory, references=None):
        self.directory = Path(directory).resolve()
        candidates = json.loads((self.directory/'candidate_manifest.json').read_text())
        self.candidates = {c['id']: c for c in candidates}
        if len(candidates) != len(self.candidates) or any(
                not isinstance(c['id'], str) or '/' in c['id'] or '\\' in c['id'] or c['id'] in ('.','..')
                for c in candidates):
            raise ValueError('unsafe or duplicate candidate IDs')
        # Stable membership: approved/rejected items remain browsable in this group.
        self.initial_ids = [c['id'] for c in candidates if c.get('pool') in ('identity','motion')]
        self.path = self.directory/'image_labels.json'
        payload = json.loads(self.path.read_text()) if self.path.exists() else {'labels':{}}
        self.labels = payload['labels']
        for ident, entry in self.labels.items():
            if ident not in self.candidates or entry['source_sha256'] != self.candidates[ident]['source_sha256']:
                raise ValueError('saved labels do not match current source manifest')
            if entry['label'] not in ('identity','motion','reject','hold'):
                raise ValueError('invalid saved label')
        self.references = sorted(p for p in Path(references).expanduser().glob('*')
                                 if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')) if references else []
        self.calibration_ids = []
        queue_path = self.directory/'calibration_queue.json'
        if queue_path.exists():
            queue = json.loads(queue_path.read_text())
            for ident in queue['ids']:
                if (ident not in self.candidates or ident in self.calibration_ids
                        or queue['source_hashes'].get(ident) != self.candidates[ident]['source_sha256']):
                    raise ValueError('calibration queue does not match source manifest')
                self.calibration_ids.append(ident)
        self.proposals = {}
        self._proposal_stamp = None
        self.lock = threading.RLock()
        self.refresh_proposals()
        self.frames_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)

    def refresh_proposals(self):
        proposal_path = self.directory/'quality_proposals.json'
        if not proposal_path.exists():
            return
        stamp = proposal_path.stat().st_mtime_ns
        if stamp == self._proposal_stamp:
            return
        payload = json.loads(proposal_path.read_text())
        if payload.get('cpu_only') is not True or payload.get('schema_version') != 1:
            raise ValueError('unexpected quality proposals')
        proposals = {}
        for ident, proposal in payload['proposals'].items():
            if (ident not in self.candidates or proposal['source_sha256'] != self.candidates[ident]['source_sha256']
                    or proposal['suggested_label'] not in (None,'identity','motion','reject')
                    or proposal['decision'] not in ('auto_proposal','review')
                    or proposal['needs_review'] != (proposal['decision']=='review')):
                raise ValueError('proposal does not match manifest')
            proposals[ident] = proposal
        with self.lock:
            self.proposals = proposals
            self._proposal_stamp = stamp

    def label(self, ident, label):
        if ident not in self.candidates or label not in (None,'identity','motion','reject','hold'):
            raise ValueError('unknown candidate or label')
        with self.lock:
            labels = dict(self.labels)
            if label is None:
                labels.pop(ident,None)
            else:
                c = self.candidates[ident]
                labels[ident] = {'label':label,'source_sha256':c['source_sha256'],
                    'updated_at':datetime.now(timezone.utc).isoformat(),
                    'review_scope':'sampled_color_frames', 'audio_stability':'user_reported_verified'}
            dump(self.path, {'schema_version':1,'labels':labels,
                'audio_evidence':'USER reports stable audio after accelerated listening; image labels separate',
                'full_clip_review_complete':False})
            self.labels = labels
            return dict(labels)

    def frame(self, ident):
        if ident not in self.candidates:
            raise ValueError('unknown candidate')
        c = self.candidates[ident]
        target = self.directory/'color-previews'/f"{ident}-{c['source_sha256'][:16]}.jpg"
        with self.frames_lock:
            if target.exists():
                return target
            if shutil.disk_usage(self.directory).free < 512*1024**2:
                raise OSError('less than 512 MiB free; cannot generate color preview')
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix('.partial.jpg')
            try:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(c['start_seconds']+.15),
                    '-i',c['source'],'-t',str(c['end_seconds']-c['start_seconds']-.3),'-an',
                    '-vf','fps=1,scale=320:-2,tile=5x1','-frames:v','1','-update','1',
                    str(temporary)],check=True,capture_output=True,timeout=40)
                if not temporary.exists() or not temporary.stat().st_size:
                    raise OSError('empty preview')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target


def make_server(store, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def send(self, body, kind='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.valid_host():
                return self.send(b'bad host','text/plain',403)
            path = urlparse(self.path).path
            try:
                if path == '/':
                    return self.send(PAGE.replace('__TOKEN__',json.dumps(store.token)).encode(),'text/html; charset=utf-8')
                if path == '/data':
                    store.refresh_proposals()
                    with store.lock:
                        body={'candidates':list(store.candidates.values()),'labels':dict(store.labels),
                              'references':[p.name for p in store.references], 'calibration_ids':store.calibration_ids,
                              'initial_ids':store.initial_ids,'proposals':store.proposals}
                    return self.send(json.dumps(body,ensure_ascii=False).encode())
                if path.startswith('/frame/'):
                    return self.send(store.frame(path.removeprefix('/frame/')).read_bytes(),'image/jpeg')
                if path.startswith('/reference/'):
                    index=int(path.removeprefix('/reference/'))
                    if index<0 or index>=len(store.references):
                        raise ValueError('reference not found')
                    file=store.references[index]
                    return self.send(file.read_bytes(),'image/png' if file.suffix.lower()=='.png' else 'image/jpeg')
                return self.send(b'not found','text/plain',404)
            except (ValueError,KeyError):
                self.send(b'not found','text/plain',404)
            except (OSError,subprocess.SubprocessError) as exc:
                self.log_error('preview failed: %s',exc)
                self.send(b'preview unavailable','text/plain',500)

        def do_POST(self):
            origin=self.headers.get('Origin')
            allowed=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
            if (not self.valid_host() or self.headers.get('X-Review-Token')!=store.token
                    or (origin is not None and origin not in allowed)):
                return self.send(b'forbidden','text/plain',403)
            if self.path!='/label':
                return self.send(b'not found','text/plain',404)
            try:
                if self.headers.get('Content-Type')!='application/json':
                    raise ValueError('JSON required')
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>4096:
                    raise ValueError('invalid request size')
                d=json.loads(self.rfile.read(length))
                self.send(json.dumps({'labels':store.label(d['id'],d['label'])}).encode())
            except (ValueError,KeyError,TypeError):
                self.send(b'invalid label request','text/plain',400)
            except OSError:
                self.send(b'save failed','text/plain',500)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    store=ReviewStore(a.selection_dir,a.reference_dir)
    server=make_server(store,a.port)
    print(f'Local image review: http://127.0.0.1:{server.server_port} (Ctrl-C stops)',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
