"""Model-free, unpinned, finite 2H200 location probe. No HF credentials/assets."""
import argparse
from decimal import Decimal
import fcntl
import inspect
import json
import os
from pathlib import Path
import time
import uuid

RATE = Decimal('9.134352')  # 2H200 + hard1 physical CPU + hard1GiB, no regional premium


def classify(metadata):
    region = metadata.get('MODAL_REGION', '')
    cloud = metadata.get('MODAL_CLOUD_PROVIDER', '')
    known_clouds = ('CLOUD_PROVIDER_AWS','CLOUD_PROVIDER_GCP','CLOUD_PROVIDER_OCI')
    countries = {'jp':'JP','au':'AU','ca':'CA','mx':'MX'}
    if cloud not in known_clouds:
        return {'decision':'unknown','country':None}
    if region in countries:
        return {'decision':'allowed_candidate','country':countries[region]}
    if region in ('us','eu','uk') or region.startswith(('us-','eu-')):
        return {'decision':'excluded','country':None}
    return {'decision':'unknown','country':None}


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.chmod(0o600);os.replace(temp,path)


def build_probe():
    import modal
    classifier_source = inspect.getsource(classify)
    app = modal.App('h3-model-free-location-'+uuid.uuid4().hex[:10])
    @app.function(image=modal.Image.debian_slim(python_version='3.11'),
                  gpu='H200:2',cpu=(1,1),memory=(1024,1024),
                  timeout=15,startup_timeout=45,scaledown_window=2,
                  single_use_containers=True,retries=0,max_containers=1,
                  serialized=True,include_source=False,block_network=True,
                  restrict_modal_access=True)
    def probe():
        import os
        import subprocess
        keys=('MODAL_REGION','MODAL_CLOUD_PROVIDER','MODAL_TASK_ID')
        metadata={k:os.environ.get(k,'') for k in keys}
        namespace={};exec(classifier_source,namespace)
        decision=namespace['classify'](metadata)
        try:
            proc=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                                capture_output=True,text=True,timeout=5)
            gpu_names=proc.stdout.strip().splitlines() if proc.returncode==0 else []
        except (OSError,subprocess.TimeoutExpired):
            gpu_names=[]
        return dict(metadata=metadata,classification=decision,gpu_names=gpu_names,
                    requested_gpu='H200:2',model_downloaded=False,model_loaded=False)
    return app,probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    plan=dict(max_attempts=3,rate_usd_hour=str(RATE),function_seconds=15,
              startup_seconds=45,scaledown_seconds=2,reservation_usd='1.00',
              modal_pool_usd='20.44',after_reservation_usd='19.44',
              region_specified=False,model_free=True,not_platform_billing_cap=True)
    if not a.run:print(json.dumps(plan));return 0
    root=Path(__file__).resolve().parents[1]
    out=a.output.resolve();budget=a.budget.resolve()
    if out.exists() or out.is_relative_to(root) or budget.is_relative_to(root):raise ValueError('fresh private output/budget required')
    out.mkdir(parents=True,mode=0o700);budget.parent.mkdir(parents=True,exist_ok=True)
    key=uuid.uuid4().hex
    with budget.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(budget.read_text()) if budget.exists() else dict(schema='h3-ref2va-modal-budget-v1',cap_usd='20.44',spent_bound_usd='0',reservations={})
        if data['schema']!='h3-ref2va-modal-budget-v1' or Decimal(data['cap_usd'])!=Decimal('20.44'):raise ValueError('wrong Modal authority')
        used=Decimal(data['spent_bound_usd'])+sum(Decimal(v['upper_bound_usd']) for v in data['reservations'].values())
        if data['reservations'] or used+1>Decimal('20.44'):raise ValueError('active reservation or insufficient Modal pool')
        data['reservations'][key]=dict(upper_bound_usd='1.00',scope='model-free-country-probe',output=str(out))
        save(budget,data)
    save(out/'plan.json',plan);save(out/'run-started.json',dict(key=key,source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
    observations=[];started=time.time()
    try:
        import modal
        app,probe=build_probe()
        with modal.enable_output():
            with app.run():
                for index in range(3):
                    row=probe.remote();row['attempt']=index+1
                    observations.append(row);save(out/'observations.json',observations)
                    print('LOCATION_PROBE',json.dumps(row),flush=True)
                    # Unknown metadata is not evidence that another paid draw will help.
                    if row['classification']['decision']!='excluded':break
                    task=row['metadata'].get('MODAL_TASK_ID')
                    if not task or sum(r['metadata'].get('MODAL_TASK_ID')==task for r in observations)>1:break
    finally:
        # Conservative charge of the entire phase allowance, NOT an invoice claim.
        with budget.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);data=json.loads(budget.read_text())
            data['reservations'].pop(key)
            data['spent_bound_usd']=str(Decimal(data['spent_bound_usd'])+1)
            save(budget,data)
        save(out/'terminal.json',dict(observations=len(observations),elapsed_wall_seconds=time.time()-started,
            phase_accounting_bound_usd='1.00',invoice_verified=False,includes_build_allowance=True,
            provider_billing_not_verified=True,modal_pool_remaining_bound_usd=str(Decimal('20.44')-Decimal(data['spent_bound_usd'])),
            no_model_execution=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
