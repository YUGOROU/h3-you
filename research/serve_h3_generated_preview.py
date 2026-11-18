"""Loopback-only, manual-play preview of SHA-verified generated receive segments.

Never opens a browser/OBS, captures a device, routes audio, or connects Discord.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Timer

HTML=b'''<!doctype html><meta charset="utf-8"><title>H3 generated-only preview</title>
<style>body{background:#111;color:white;font:16px sans-serif}video{height:80vh;max-width:100%;background:black}</style>
<p>Generated silent video only. Manual play. No camera, microphone or screen capture.</p>
<div id="choices"></div><p id="status">Waiting for verified segments...</p>
<video id="v" controls muted playsinline preload="none"></video>
<script>
async function refresh(){const rows=await(await fetch('/status')).json();
document.getElementById('status').textContent=rows.length+' verified segments; not continuous live playback';
const c=document.getElementById('choices');c.replaceChildren();
for(const row of rows){let b=document.createElement('button');b.textContent=row;
b.onclick=()=>{const v=document.getElementById('v');v.src='/'+row;
v.onloadedmetadata=()=>{v.currentTime=Number(row.slice(8,11))*124/24;};};c.appendChild(b);}}
refresh();setInterval(refresh,2000);
</script>'''


def verified(root):
    receipt=root/'receive-receipt.json'
    if not receipt.is_file() or receipt.is_symlink():return {}
    try:rows=json.loads(receipt.read_text())['segments']
    except (ValueError,KeyError):return {}  # Atomic receipt replacement may be in flight.
    return {name:row for name,row in rows.items() if re.fullmatch(r'segment-00[0-3]\.mp4',name) and row.get('pts_full_decode_verified') is True}


def handler(root):
    root=Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':self.send_error(403);return
            path=self.path
            if path=='/':body,kind=HTML,'text/html; charset=utf-8'
            elif path=='/status':body,kind=json.dumps(sorted(verified(root))).encode(),'application/json'
            elif re.fullmatch(r'/segment-00[0-3]\.mp4',path):
                name=path[1:];rows=verified(root);file=root/name
                if name not in rows or file.is_symlink() or not file.is_file() or file.stat().st_size>16*1024*1024:self.send_error(404);return
                body=file.read_bytes()
                if hashlib.sha256(body).hexdigest()!=rows[name]['sha256']:self.send_error(409);return
                kind='video/mp4'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cross-Origin-Resource-Policy','same-origin')
            self.send_header('Permissions-Policy','camera=(), microphone=(), display-capture=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--seconds',type=int,default=1800);a=p.parse_args()
    if not 1<=a.seconds<=1800:raise ValueError('finite preview lifetime required')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(a.root));timer=Timer(a.seconds,server.shutdown);timer.daemon=True;timer.start()
    print(f'Manual preview ready: http://127.0.0.1:{server.server_port}/ (no browser opened)',flush=True)
    try:server.serve_forever()
    finally:timer.cancel();server.server_close()

if __name__=='__main__':main()
