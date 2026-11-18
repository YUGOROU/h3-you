"""Explicit public FLEURS Japanese check within original T4 reservation/deadline.
No new Endpoint or reservation, no personal capture/upload. Always pause again.
"""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import time
import unicodedata
from urllib.parse import urlsplit
from research.run_h3_asr_endpoint import control, sha, assert_deployment, pause_owned

PUBLIC_SHA = '9627b8f5d46806e1a433b443d692fdfb24f26a35c1a47eba26291475fc643d7b'
REFERENCE = '多くの場合、海外のギャップイヤーコースに入学することで、帰国後に実際に大学に進学しやすくなります。'


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace() and not unicodedata.category(c).startswith('P'))


def cer(reference, hypothesis):
    a, b = normalized(reference), normalized(hypothesis)
    row = list(range(len(b)+1))
    for i, left in enumerate(a,1):
        new = [i]
        for j, right in enumerate(b,1):
            new.append(min(row[j]+1,new[j-1]+1,row[j-1]+(left!=right)))
        row = new
    return row[-1]/len(a) if a else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if not args.run:
        print(json.dumps(dict(public_dataset='google/fleurs',language='ja-JP',new_allocation=False,pause_after=True)))
        return
    from research.provision_modal_identity_secret import cached_token
    from research.probe_h3_asr_cpu_space import http_ready,stream_public
    root=args.directory
    parent=json.loads((root/'result.json').read_text())
    output=root/'japanese-result.json'
    if output.exists() or time.time()>parent['deadline']-600:
        raise ValueError('no replay or deadline extension permitted')
    raw=root/'japanese-public.wav'
    if sha(raw)!=PUBLIC_SHA:
        raise ValueError('exact pinned public FLEURS sample required')
    receipt=json.loads((root/'japanese-public-receipt.json').read_text())
    if receipt['reference']!=REFERENCE or receipt['revision']!='70bb2e84b976b7e960aa89f1c648e09c59f894dd':
        raise ValueError('pinned Japanese reference/revision required')
    pcm=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(raw),
                        '-ar','16000','-ac','1','-f','s16le','pipe:1'],check=True,capture_output=True,timeout=15).stdout
    if len(pcm)!=355200:
        raise ValueError('exact public duration11.1s required')
    token=cached_token();path='/v2/endpoint/'+parent['namespace']+'/'+parent['name']
    current=control(token,path)
    expected=json.loads((root/'intent.json').read_text())
    assert_deployment(current,expected)
    if current['status']['state']!='paused' or current['model']['revision']!=parent['revision']:
        raise ValueError('same paused immutable Endpoint required')
    if expected['model']['env']['H3_ASR_EXPIRES_AT']!=str(parent['deadline']):
        raise ValueError('original deadline required')
    ledger=json.loads((root.parent/'compute-budget.json').read_text())
    reservation=ledger['reservations'].get(parent['reservation'],{})
    if reservation.get('scope')!='hf-endpoint-ASR-T4-public-validation' or reservation.get('max_seconds')!=1800 or reservation.get('max_replicas')!=3 or reservation.get('upper_bound_usd')!=0.75:
        raise ValueError('original locked reservation required')
    key=secrets.token_urlsafe(40)
    result=dict(name=parent['name'],reservation=parent['reservation'],original_deadline=parent['deadline'],
                public_sample_only=True,reference=REFERENCE,tests=[])
    try:
        # Rotate ONLY owner stream key; no model/provider/deadline change.
        control(token,path,dict(model=dict(secrets=dict(H3_ASR_KEY=key))),'PUT')
        current=control(token,path)
        if current['status']['state']=='paused':
            control(token,path+'/resume',{},'POST')
        end=time.monotonic()+240
        while True:
            current=control(token,path);state=current['status']['state']
            print(json.dumps(dict(name=parent['name'],state=state)),flush=True)
            if state=='running':break
            if state=='failed' or time.monotonic()>end:raise RuntimeError('finite resume readiness failed')
            time.sleep(10)
        parsed=urlsplit(current['status']['url'])
        if parsed.scheme!='https' or not parsed.hostname.endswith('.endpoints.huggingface.cloud'):
            raise ValueError('HF host required')
        anon,_=http_ready(parsed.hostname)
        code,body=http_ready(parsed.hostname,token)
        if anon not in (401,403,404) or code!=200 or json.loads(body).get('device')!='cuda':
            raise ValueError('anonymous denied/authenticated CUDA readiness required')
        for repeats in (1,3):
            test=stream_public(parsed.hostname,token,pcm*repeats,16000,{'X-H3-ASR-Key':key},language='ja-JP')
            test['normalized_cer']=cer(REFERENCE*repeats,''.join(test['final_segments']))
            test['last_final_relative_to_wav_end_seconds']=test['last_final_seconds']-test['audio_seconds'] if test['last_final_seconds'] is not None else None
            result['tests'].append(test)
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps({k:v for k,v in test.items() if k not in ('events','final_segments')},ensure_ascii=False),flush=True)
            if not test['final_received'] or test['errors']:raise ValueError('Japanese complete result required')
    finally:
        result['terminal_state']='unconfirmed'
        try:
            result['terminal_state']=pause_owned(token,path)
            print(json.dumps(dict(name=parent['name'],terminal_state=result['terminal_state'])),flush=True)
        finally:
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
