"""Mac-side isolated RTC child owner. No capture, app launch or endpoint resume."""
from dataclasses import asdict
import os
import math
from pathlib import Path
import re
import socket
import subprocess
import time
from .webrtc_sidecar import clean_env
from .webrtc_exact_channel import Binding
from .webrtc_upstream_adapter import RTCStream,asr_transport
from . import paired_av_wire as wire

ROOT=Path(__file__).resolve().parents[2]

class ASRRTCFactory:
    media_transport='webrtc-exact-v1'
    def __init__(self,python=None):
        self.python=ROOT/'research/webrtc_experiment/.venv/bin/python' if python is None else Path(python)
        if not self.python.is_file():raise ValueError('existing approved isolated RTC Python required')
        self._active=None

    def open_turn(self,*,endpoint,token,key,session,epoch,expires_at,deadline):
        from .endpoint_asr import _host,_credential
        _host(endpoint);_credential(token);_credential(key)
        Binding(session,'validate',epoch,'asr-native-event-wire-v1',4*1024*1024)
        if (type(deadline) not in (int,float) or not math.isfinite(deadline) or not time.monotonic()<deadline<=time.monotonic()+120
                or type(expires_at) not in (int,float) or not math.isfinite(expires_at) or not time.time()<expires_at<=time.time()+1800):
            raise ValueError('finite exact RTC turn deadline required')
        if self._active is not None:raise RuntimeError('serial ASR RTC owner required')
        turn=Turn(self,deadline);self._active=turn
        try:
            turn.start(endpoint,token,key,session,epoch,expires_at)
            return turn
        except BaseException as primary:
            try:turn.close(require_complete=False)
            except BaseException:primary.add_note('RTC child cleanup unconfirmed')
            raise

class Turn:
    def __init__(self,owner,deadline):
        self.owner=owner;self.deadline=deadline;self.process=None;self.control=None;self.media=None
        self.transport=None;self.receipt=None;self._closed=False
    def start(self,endpoint,token,key,session,epoch,expires_at):
        controls=socket.socketpair();media=socket.socketpair()
        self.control=controls[0];self.media=media[0]
        try:
            env=dict(clean_env(),PYTHONPATH=str(ROOT),H3_ASR_RTC_TOKEN=token,H3_ASR_RTC_KEY=key)
            self.process=subprocess.Popen([str(self.owner.python),'-m','research.h3_personal.asr_rtc_client',
                str(controls[1].fileno()),str(media[1].fileno())],pass_fds=(controls[1].fileno(),media[1].fileno()),
                env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            # Only the child may retain these endpoints; otherwise child exit
            # cannot produce EOF while the parent waits for readiness.
            controls[1].close();media[1].close()
            wire._write(self.control,wire._pack(dict(endpoint=endpoint,session=session,epoch=epoch,
                expires_at=expires_at,deadline=self.deadline)),self.deadline)
            row=wire._json(self.control,self.deadline)
            from .asr_rtc_client import STAGES
            if (set(row)=={'schema','stage'} and row['schema']=='h3-asr-rtc-ipc-error-v1'
                    and type(row['stage']) is str and row['stage'] in STAGES):
                raise RuntimeError('RTC child failed at stage: '+row['stage'])
            if type(row) is not dict or set(row)!={'schema','binding'} or row['schema']!='h3-asr-rtc-ipc-ready-v1':raise ValueError('RTC child readiness')
            binding=Binding(**row['binding'])
            if (binding.session!=session or binding.epoch!=epoch or binding.purpose!='asr-native-event-wire-v1'
                    or binding.max_bytes!=4*1024*1024 or not re.fullmatch('[a-f0-9]{32}',binding.boot)):
                raise ValueError('RTC child binding mismatch')
            self.transport=asr_transport(RTCStream(self.media,binding))
        finally:controls[1].close();media[1].close()
    def close(self,*,require_complete=True):
        if self._closed:return
        self._closed=True;error=None
        try:
            if self.media is not None:
                try:self.media.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                self.media.close()
            if require_complete:
                self.receipt=wire._json(self.control,min(self.deadline,time.monotonic()+3))
                expected={'schema','bytes','sha256','transport_complete','native_eos_verified','local_eof_already_closed'}
                r=self.receipt
                from .asr_rtc_client import STAGES
                if (type(r) is dict and set(r)=={'schema','stage'}
                        and r['schema']=='h3-asr-rtc-ipc-error-v1'
                        and type(r['stage']) is str and r['stage'] in STAGES):
                    raise RuntimeError('RTC child failed at stage: '+r['stage'])
                if (type(r) is not dict or set(r)!=expected or r['schema']!='h3-asr-rtc-ipc-complete-v1'
                        or r['transport_complete'] is not True or r['native_eos_verified'] is not False
                        or type(r['bytes']) is not int or not 0<=r['bytes']<=4*1024*1024
                        or type(r['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',r['sha256'])
                        or type(r['local_eof_already_closed']) is not bool):raise ValueError('RTC carrier completion required')
                self.process.wait(timeout=2)
                if self.process.returncode!=0:raise RuntimeError('RTC child failed')
        except BaseException as caught:error=caught
        finally:
            if self.control is not None:self.control.close()
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            self.owner._active=None
        if error is not None:raise error
