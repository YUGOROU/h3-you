"""Finite isolated L4 candidate worker; no H3/device/playback or media upload."""
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare_compiler(spec):
    """Compile-only toolchain in disposable Job, never the Mac/model runtime."""
    optimize=validate_profile(spec)
    if not (optimize or spec.get('voxcpm_lm_compile',False)):
        return
    cc = shutil.which('cc') or shutil.which('gcc')
    if cc is None:
        # The pinned torch runtime image has no C compiler. Triton needs one
        # even though model weights and all Python runtime pins are unchanged.
        if os.geteuid() != 0 or not shutil.which('apt-get'):
            raise RuntimeError('compile profile requires a C compiler')
        env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
        subprocess.run(['apt-get', 'update', '-qq'], env=env, check=True, timeout=90)
        subprocess.run(['apt-get', 'install', '-y', '--no-install-recommends', 'gcc', 'libc6-dev'],
                       env=env, check=True, timeout=120)
        cc = shutil.which('gcc')
    if not cc:
        raise RuntimeError('C compiler installation incomplete')
    os.environ['CC'] = cc
    version = subprocess.run([cc, '--version'], check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    (Path(spec['_compiler_receipt_root']) / 'compiler.json').write_text(json.dumps({'cc':cc,'version':version}))


def bootstrap(root, spec, reexec_target=None):
    """uv from hash-pinned wheel; compile-only disposable OS toolchain."""
    compiler_spec = dict(spec, _compiler_receipt_root=str(root))
    prepare_compiler(compiler_spec)
    uvdir=Path('/tmp/h3-probe-uv');uvdir.mkdir(mode=0o700)
    wheel=uvdir/'uv.whl'
    with urllib.request.urlopen(spec['uv']['url'],timeout=60) as r, wheel.open('wb') as f:
        while block:=r.read(1024*1024):f.write(block)
    if sha(wheel)!=spec['uv']['sha256']:raise ValueError('uv bootstrap hash mismatch')
    with zipfile.ZipFile(wheel) as z:
        matches=[n for n in z.namelist() if n.endswith('/uv')]
        if len(matches)!=1:raise ValueError('uv executable inventory')
        uv=uvdir/'uv';uv.write_bytes(z.read(matches[0]));uv.chmod(0o700)
    env=dict(os.environ,UV_NO_CACHE='1',SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM='0.0.0+git.cce58a5')
    if spec['candidate']=='voxcpm2':
        project=root/'research/tts_streaming_experiment/voxcpm2'
        req=Path('/tmp/h3-probe-locked-requirements.txt')
        subprocess.run([str(uv),'export','--project',str(project),'--frozen','--no-dev','--no-emit-project','--output-file',str(req)],env=env,check=True,timeout=90,stdout=subprocess.DEVNULL)
        venv=Path('/tmp/h3-voxcpm-probe-env')
        subprocess.run([str(uv),'venv','--system-site-packages','--python',sys.executable,str(venv)],env=env,check=True,timeout=30)
        python=venv/'bin/python'
        subprocess.run([str(uv),'pip','install','--python',str(python),'--require-hashes','-r',str(req)],env=env,check=True,timeout=600)
    else:
        target=Path('/tmp/h3-qwen-probe-client')
        subprocess.run([str(uv),'pip','install','--target',str(target),'websockets==16.0'],env=env,check=True,timeout=90)
        env['PYTHONPATH']=str(target)+os.pathsep+str(root)
        python=Path(sys.executable)
    os.execve(str(python),[str(python),str(reexec_target if reexec_target is not None else root/'research/hf_h3_native_tts_probe.py'),str(root),'--native'],env)


def models(spec):
    from huggingface_hub import snapshot_download,hf_hub_download
    token=os.environ['HF_TOKEN']
    model=spec['model']
    path=Path(snapshot_download(model['repo'],revision=model['revision'],token=token))
    for row in model['files']:
        p=path/row['path']
        if p.stat().st_size!=row['bytes']:raise ValueError('model size mismatch')
        digest=sha(p) if row['lfs'] else hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()
        if digest!=row['digest']:raise ValueError('model bytes mismatch')
    ref=spec['reference']
    reference=Path(hf_hub_download(ref['repo'],'reference.wav',repo_type='dataset',revision=ref['revision'],token=token))
    if sha(reference)!=ref['sha256']:raise ValueError('reference drift')
    os.environ.pop('HF_TOKEN',None)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    return path,reference


class DeviceMemory:
    """Bounded 50ms NVML observation of whole device, NOT an exact allocator peak."""
    def __init__(self):
        import ctypes as c
        self.c=c;self.lib=c.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML init')
        self.handle=c.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0,c.byref(self.handle))!=0:raise RuntimeError('NVML handle')
        class Memory(c.Structure):_fields_=[('total',c.c_ulonglong),('free',c.c_ulonglong),('used',c.c_ulonglong)]
        self.Memory=Memory;self.peak=0;self.count=0;self.error=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        while not self.stop.is_set():
            data=self.Memory()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle,self.c.byref(data))!=0:
                self.error='nvml-query-failed';return
            self.peak=max(self.peak,int(data.used));self.count+=1
            self.stop.wait(.05)
    def close(self):
        self.stop.set();self.thread.join(timeout=2);self.lib.nvmlShutdown()
    def result(self):
        return dict(observed_device_used_peak_bytes=self.peak,observations=self.count,
                    poll_seconds=.05,error=self.error,scope='whole-exclusive-device-observed-not-exact-transient-peak')


def validate_profile(spec):
    optimize=spec.get('voxcpm_optimize',False)
    if type(optimize) is not bool or (optimize and spec['candidate']!='voxcpm2'):
        raise ValueError('Vox optimize profile requires VoxCPM2 and boolean option')
    abba=spec.get('voxcpm_trace_abba',False)
    if type(abba) is not bool or (abba and (spec['candidate']!='voxcpm2' or not optimize)):
        raise ValueError('Vox trace ABBA requires VoxCPM2 optimize profile')
    diagnostic=spec.get('voxcpm_supply_diagnostic',False)
    if type(diagnostic) is not bool or (diagnostic and (spec['candidate']!='voxcpm2' or optimize or abba)):
        raise ValueError('supply diagnostic requires eager VoxCPM2 only')
    stage=spec.get('voxcpm_stage_diagnostic',False)
    if type(stage) is not bool or (stage and (spec['candidate']!='voxcpm2' or optimize or abba or diagnostic)):
        raise ValueError('stage diagnostic requires eager VoxCPM2 without other experiments')
    lm=spec.get('voxcpm_lm_compile',False)
    if type(lm) is not bool or (lm and not stage):
        raise ValueError('LM-only compile requires stage diagnostic')
    return optimize


def trace_abba(candidate, text, out, kind, torch, measure):
    """Same resident model; bounded full drains, no retained warmup media."""
    import tempfile
    def call(trace, directory):
        candidate.observe_stop=trace
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        row=measure(candidate,text,directory,kind)
        row.update(observe_stop=trace,seed=42,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   matched_text=True,profile='voxcpm-optimize-trace-abba',
                   formal_completion_expected=trace,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return row
    warmups=[]
    for trace in (False,True):
        with tempfile.TemporaryDirectory(prefix='h3-trace-warmup-') as directory:
            row=call(trace,Path(directory))
            row.update(measured=False,artifacts_retained=False)
            warmups.append(row)
    rows=[]
    for i,trace in enumerate((False,True,True,False)):
        row=call(trace,out/str(i))
        row.update(case=i,measured=True,rtf=row['iterator_completion_seconds']/row['duration_seconds'],
                   memory_scope='native-process')
        rows.append(row)
    return warmups,rows


def run(root,spec):
    optimize=validate_profile(spec)
    os.umask(0o077)
    key=os.environ['H3_PROBE_KEY'];boot=os.urandom(16).hex()
    out=Path('/tmp/h3-native-tts-probe-results');out.mkdir(mode=0o700)
    status={'state':'preparing','boot':boot,'candidate':spec['candidate']}
    ack=threading.Event();artifacts={}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            self.connection.settimeout(10)
            if not hmac.compare_digest(self.headers.get('X-H3-Probe-Key',''),key):
                self.send_error(403);return
            if self.path=='/status':body=json.dumps(status,allow_nan=False).encode()
            elif self.path in artifacts:body=artifacts[self.path].read_bytes()
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-H3-Probe-Boot',boot)
            self.send_header('Connection','close');self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.path!='/ack' or not hmac.compare_digest(self.headers.get('X-H3-Probe-Key',''),key):
                self.send_error(403);return
            self.send_response(200);self.send_header('Content-Length','0');self.end_headers();ack.set()
    server=ThreadingHTTPServer(('0.0.0.0',7860),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    failure=None;memory=None
    try:
        import torch
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA L4':
            raise ValueError('exactly one actual L4 required')
        status['device']={'name':torch.cuda.get_device_name(0),'count':1,'total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__}
        memory=DeviceMemory()
        model,reference=models(spec)
        status['state']='loading'
        if spec['candidate']=='voxcpm2':
            from research.h3_personal.voxcpm2_streaming_candidate import load
            candidate=load(model,reference,optimize=optimize);kind='float'
            import contextlib
            ctx=contextlib.nullcontext()
        else:
            from research.h3_personal.qwen3_streaming_candidate import Qwen3StreamingCandidate
            from research.h3_personal import qwen3_streaming_runtime as runtime
            import vllm_omni
            source=Path(vllm_omni.__file__).resolve().parent.parent
            runtime.verify_install(source,version=importlib.metadata.version)
            import yaml
            config=yaml.safe_load((source/'vllm_omni/deploy/qwen3_tts.yaml').read_text())
            # Explicit revised L4 profile: official CUDA V2/eager, single utterance,
            # BF16, unchanged async PCM connector/context. Never an OOM fallback.
            config['model_runner']='v2'
            config['connectors']['connector_of_shared_memory']['extra']['codec_chunk_ramp']=[4,4,8,16,25]
            for stage in config['stages']:
                stage.update(max_num_seqs=1,enforce_eager=True,dtype='bfloat16')
            config_path=out/'l4-config.yaml';config_path.write_text(yaml.safe_dump(config))
            reference_dir=Path('/tmp/h3-qwen-private-reference');reference_dir.mkdir(mode=0o700)
            local_reference=reference_dir/'reference.wav';shutil.copyfile(reference,local_reference)
            if sha(local_reference)!=spec['reference']['sha256']:raise ValueError('reference copy drift')
            ctx=runtime.server(model,config_path,accepted_revision=runtime.RELEASE_REVISION,
                               accepted_bf16=True,lifetime=600,readiness=300,reference_directory=reference_dir)
            candidate=Qwen3StreamingCandidate(local_reference,x_vector_only=True,reference_transport='local-file',
                expected_reference_sha256=spec['reference']['sha256'],max_new_tokens=512)
            kind='pcm16'
        result={'candidate':spec['candidate'],'model_revision':spec['model']['revision'],
                'device':status['device'],'reference_sha256':spec['reference']['sha256'],
                'h3_integrated':False,'playback_verified':False,'watermark':'not-Irodori; no watermark claim',
                'cloning_mode':'reference-only' if kind=='float' else 'explicit-x-vector-only-no-transcript',
                'voxcpm_optimize':optimize,'voxcpm_trace_abba':spec.get('voxcpm_trace_abba',False),'tests':[]}
        if optimize or spec.get('voxcpm_lm_compile',False):
            result['compiler']=json.loads((root/'compiler.json').read_text())
        from research.h3_personal.tts_streaming_measure import measure
        with ctx:
            status['state']='synthesizing'
            if spec.get('voxcpm_stage_diagnostic',False):
                from research.h3_personal.voxcpm_stage_diagnostic import run as profile_stages
                candidate.observe_stop=False
                result['stage_diagnostic']=profile_stages(candidate,spec['texts'][0],out/'stages',torch,measure,lm_compile=spec.get('voxcpm_lm_compile',False))
                result['tests']=[case['measurement'] for case in result['stage_diagnostic']['cases']]
            if spec.get('voxcpm_supply_diagnostic',False):
                from research.h3_personal.voxcpm_supply_diagnostic import run as diagnose
                candidate.observe_stop=False
                result['runtime_threads']={'intra':torch.get_num_threads(),'inter':torch.get_num_interop_threads()}
                result['supply_diagnostic']=diagnose(candidate,spec['texts'][0],out/'diagnostic',torch,measure)
                result['tests']=[case['measurement'] for case in result['supply_diagnostic']['cases']]
            if spec.get('voxcpm_trace_abba',False):
                result['warmups'],result['tests']=trace_abba(candidate,spec['texts'][0],out,kind,torch,measure)
            for i,text in enumerate([] if (spec.get('voxcpm_trace_abba',False) or spec.get('voxcpm_supply_diagnostic',False) or spec.get('voxcpm_stage_diagnostic',False)) else spec['texts']):
                torch.cuda.reset_peak_memory_stats()
                row=measure(candidate,text,out/str(i),kind)
                row['case']=i;row['text_sha256']=hashlib.sha256(text.encode()).hexdigest()
                row['rtf']=row['iterator_completion_seconds']/row['duration_seconds']
                row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                row['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                row['memory_scope']='native-process' if kind=='float' else 'client-only-not-vllm-server'
                result['tests'].append(row)
                print(json.dumps(dict(type='tts_candidate_case',candidate=spec['candidate'],case=i,
                    ttfa=row['ttfa_seconds'],seconds=row['duration_seconds'],chunks=row['chunk_count'],
                    rtf=row['rtf'],late_supply=row['max_late_supply_seconds'])),flush=True)
        result['formal_completion_accepted']=all(row['formal_completion_accepted'] for row in result['tests'])
        if spec.get('voxcpm_trace_abba',False) or spec.get('voxcpm_supply_diagnostic',False) or spec.get('voxcpm_stage_diagnostic',False):
            result['formal_completion_note']='Untraced cases intentionally have unknown completion; formal acceptance false is expected, not native EOS evidence.'
        memory.close();result['device_memory']=memory.result();memory=None
        result_path=out/'result.json';result_path.write_text(json.dumps(result,indent=2,allow_nan=False))
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.wav','.yaml'):
                name='/artifacts/'+str(p.relative_to(out));artifacts[name]=p
        manifest=[{'path':n,'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in artifacts.items()]
        if sum(r['bytes'] for r in manifest)>32*1024*1024:raise ValueError('artifact budget')
        status.update(state='complete',artifacts=manifest)
    except BaseException as error:
        failure=error
        status.update(state='failed',error_type=type(error).__name__)
        if 'candidate' in locals() and getattr(candidate,'last_failure',None):
            status['native_failure']=candidate.last_failure
        print(json.dumps(dict(type='tts_candidate_failed',candidate=spec['candidate'],error_type=type(error).__name__)),flush=True)
    finally:
        if memory is not None:memory.close()
        ack.wait(timeout=max(0,min(120,spec['expires_at']-time.time()-10)))
        server.shutdown();server.server_close();thread.join(timeout=2)
    if failure is not None:raise failure


if __name__=='__main__':
    root=Path(sys.argv[1]);sys.path.insert(0,str(root));os.chdir(root)
    spec=json.loads((root/'request.json').read_text())
    validate_profile(spec)
    os.umask(0o077)
    remaining=int(spec['expires_at']-time.time())
    if not 0<remaining<=1200:raise ValueError('expired finite worker')
    signal.alarm(remaining)
    if len(sys.argv)<3:bootstrap(root,spec)
    else:run(root,spec)
