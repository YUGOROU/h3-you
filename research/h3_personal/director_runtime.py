"""Token-safe Pi child configuration and bounded HF Router accounting."""
from contextlib import contextmanager
from decimal import Decimal,ROUND_CEILING
import fcntl
import json
import os
from pathlib import Path
import uuid

MODEL='Qwen/Qwen3.8-27B:cerebras'
TURN_BOUND=Decimal('.20')  # <=3 calls * (65536 input*.99 +1024 output*1.49)/1M
CALL_BOUND=Decimal('.0664064')


def child_environment(token,max_turns,*,long_session=False):
    if type(long_session) is not bool or not 1<=max_turns<=(1000 if long_session else 8):raise ValueError('finite turn limit required')
    env={key:os.environ[key] for key in ('PATH','HOME','LANG','TMPDIR') if key in os.environ}
    env.update(HF_TOKEN=token,H3_DIRECTOR_MAX_TURNS=str(max_turns))
    if long_session:env['H3_DIRECTOR_LONG_SESSION']='1'
    return env


def atomic(path,data):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with os.fdopen(os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as stream:
        json.dump(data,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


@contextmanager
def locked(path):
    path=Path(path)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);yield json.loads(path.read_text())


def reserve(path,max_turns):
    from research.h3_personal.ref2va_budget import HF_CAP,ORIGIN
    if not 1<=max_turns<=8:raise ValueError('finite turn limit required')
    bound=TURN_BOUND*max_turns;key='h3-director-'+uuid.uuid4().hex
    with locked(path) as data:
        total=Decimal(str(data['spent_usd']))+sum(Decimal(str(row['upper_bound_usd'])) for row in data['reservations'].values())
        if not total.is_finite() or total<ORIGIN or total+bound>HF_CAP:raise ValueError('HF authority exceeded')
        data['reservations'][key]=dict(provider='hf-inference',model=MODEL,upper_bound_usd=str(bound),max_turns=max_turns,kind='finite-director')
        atomic(path,data)
    return key,bound


def reconcile(path,key,usage):
    with locked(path) as data:
        row=data['reservations'][key];bound=Decimal(row['upper_bound_usd'])
        charged=bound
        if usage is not None:
            values=[usage.get(k) for k in ('calls','unknown_calls','input','output')]
            if all(type(n) is int and n>=0 for n in values) and values[0]<=3*row['max_turns'] and values[1]<=values[0]:
                calls,unknown,inputs,outputs=values
                known=Decimal(inputs)*Decimal('.99')/1000000+Decimal(outputs)*Decimal('1.49')/1000000
                charged=(known+unknown*CALL_BOUND).quantize(Decimal('.01'),rounding=ROUND_CEILING)
        data['reservations'].pop(key);data['spent_usd']=float(Decimal(str(data['spent_usd']))+charged);atomic(path,data)
    return dict(reservation=key,runtime_bound_usd=str(charged),reserved_usd=str(bound),bound_exceeded=charged>bound,usage=usage,invoice_verified=False)
