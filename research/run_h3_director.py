"""Minimal text-only Pi director. Default checks locally; --run permits HF inference.

No shell expansion of credentials, persisted Pi transcripts, media playback or devices.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import uuid
from research.h3_personal.director_runtime import MODEL,child_environment,reserve,reconcile,atomic
from research.h3_personal.private_root import root as _private_root

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=_private_root()


def verify_catalog(token):
    request=urllib.request.Request('https://router.huggingface.co/v1/models',headers={'Authorization':'Bearer '+token})
    try:
        with urllib.request.urlopen(request,timeout=15) as response:data=json.load(response)
        model=next(r for r in data['data'] if r['id']=='Qwen/Qwen3.8-27B')
        route=next(r for r in model['providers'] if r['provider']=='cerebras')
        if route['status']!='live' or route['context_length']!=65536 or route.get('supports_tools') is not True or route['pricing']!={'input':.99,'output':1.49}:raise ValueError('catalog differs')
    except Exception:raise RuntimeError('exact Cerebras route/pricing unavailable; no fallback') from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--text');p.add_argument('--serve',action='store_true');p.add_argument('--max-turns',type=int,default=1)
    a=p.parse_args(argv)
    if not 1<=a.max_turns<=8 or (a.text is not None and a.serve):p.error('choose text or serve, max-turns1..8')
    if a.run and a.text is None and not a.serve:p.error('--run needs --text or --serve')
    if a.text is not None and len(a.text.encode())>2000:p.error('text too large')
    node=shutil.which('node')
    if not node:raise RuntimeError('existing Node.js22+ required; no automatic install')
    token='offline-check'
    if a.run:
        try:token=(Path.home()/'.cache/huggingface/token').read_text().strip()
        except OSError:raise RuntimeError('HF token unavailable') from None
        if not token or '\n' in token or len(token)>2048:raise RuntimeError('invalid HF token')
        verify_catalog(token)
    command=[node,str(ROOT/'research/director/agent.mjs')];env=child_environment(token,a.max_turns)
    ledger=PRIVATE/'compute-budget.json';key=None;usage=None;child=None;returned_turn=False
    if a.run:key,_=reserve(ledger,a.max_turns)
    try:
        if a.serve and a.run:
            child=subprocess.Popen(command,stdin=sys.stdin,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,text=True,start_new_session=True)
            lines=child.stdout
        else:
            packet=json.dumps(dict(op='turn',turn_id='turn-'+uuid.uuid4().hex,text=a.text))+'\n' if a.run else ''
            completed=subprocess.run(command,input=packet,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,text=True,timeout=100)
            lines=completed.stdout.splitlines()
        for line in lines:
            try:row=json.loads(line.replace(token,'[REDACTED]'))
            except ValueError:continue
            if row.get('type')=='usage':usage=row
            if row.get('type')=='turn':returned_turn=True
            if row.get('type') in ('ready','turn','error','usage','canceled'):print(json.dumps(row,ensure_ascii=False),flush=True)
        code=child.wait(timeout=10) if child else completed.returncode
        if code:raise RuntimeError('Pi director child failed; provider details suppressed')
        if a.run and (usage is None or usage.get('turns',0)<1 or not returned_turn) and not a.serve:raise RuntimeError('no director turn executed')
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        if key:
            receipt=reconcile(ledger,key,usage);directory=PRIVATE/'director-runs';directory.mkdir(mode=0o700,exist_ok=True)
            atomic(directory/(key+'.json'),dict(model=MODEL,**receipt))
            print(json.dumps(dict(type='budget',runtime_bound_usd=receipt['runtime_bound_usd'],invoice_verified=False)),flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit(130)
    except Exception:print('Director failed; no credentials or provider payload logged.',file=sys.stderr);sys.exit(1)
