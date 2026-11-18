"""Isolated asyncio ASR RTC session manager. Native model remains private.

Expose only authenticated control/signaling through an outer HTTP gateway.
Legacy public WebSocket audio routes must not be mounted beside this manager.
"""
import asyncio
from dataclasses import asdict
import secrets
import socket
import time
from .webrtc_exact_channel import Binding,ExactChannel
from .webrtc_exact_ipc import bridge
from .webrtc_live_peer import Session
from .webrtc_signal_server import SignalServer
from .webrtc_turn import validate_turn_config
from .webrtc_ice import HF_CLOUDFLARE_TURN_TLS
from .asr_rtc_native import relay

class ASRRTCManager:
    def __init__(self,*,key,expires_at,native_ready,native_port=7861):
        if not callable(native_ready) or type(key) is not str or not 32<=len(key)<=8192:
            raise ValueError('explicit owner/native readiness required')
        if type(expires_at) not in (int,float) or not 0<expires_at-time.time()<=1800:raise ValueError('ASR lease')
        self.key=key;self.expires_at=expires_at;self.native_ready=native_ready;self.native_port=native_port
        self._session=None;self._epoch=0;self._task=None;self._signal=None;self._peer=None
        self._closed=False;self._failed=False;self._allocating=False

    async def start(self,request):
        if (self._closed or self._failed or self._allocating or self._task is not None and not self._task.done()
                or type(request) is not dict or set(request)!={'session','epoch','turn_config'}):
            raise ValueError('serial authenticated ASR RTC allocation required')
        session=request['session'];epoch=request['epoch']
        Binding(session,'validate',epoch,'asr-native-event-wire-v1',4*1024*1024)
        if type(epoch) is not int or not self._epoch<epoch<=4 or self._session not in (None,session):raise ValueError('ASR allocation replay/session')
        turn=validate_turn_config(request['turn_config'])
        if self.expires_at-time.time()<15:raise TimeoutError('ASR lease margin')
        self._allocating=True;self._epoch=epoch;self._session=session
        peer=signal=None
        try:
            ready=await self.native_ready()
            if type(ready) is not dict or ready.get('device')!='cuda':raise ValueError('native CUDA readiness required')
            if self._signal is not None:await asyncio.to_thread(self._signal.close)
            if self._peer is not None:await self._peer.close()
            seconds=min(120,self.expires_at-time.time()-1)
            expires=time.time()+seconds;deadline=time.monotonic()+seconds+.05
            peer=Session(deadline=deadline,ice_profile=HF_CLOUDFLARE_TURN_TLS,turn_config=turn)
            signal=SignalServer(session=peer,loop=asyncio.get_running_loop(),session_id=session,
                key=self.key,expires_at=expires,host='127.0.0.1',port=0)
            binding=Binding(session,signal.boot,epoch,'asr-native-event-wire-v1',4*1024*1024)
            self._peer=peer;self._signal=signal
            signal.__enter__()
            self._task=asyncio.create_task(self._serve(peer,binding))
            return dict(schema='h3-asr-rtc-turn-v1',binding=asdict(binding),expires_at=expires,
                        ice_profile=HF_CLOUDFLARE_TURN_TLS,signal_port=signal.address[1])
        except BaseException:
            self._failed=True
            if signal is not None:await asyncio.to_thread(signal.close)
            if peer is not None:await peer.close()
            raise
        finally:self._allocating=False

    async def _serve(self,peer,binding):
        channel=ExactChannel(peer,binding);native=ipc=None;tasks=[];completed=False
        try:
            await peer.wait_control();await channel.start()
            native,ipc=socket.socketpair()
            tasks=[asyncio.create_task(bridge(channel,ipc,keep_peer=True)),asyncio.create_task(asyncio.to_thread(
                relay,native,peer.deadline,port=self.native_port))]
            async with asyncio.timeout_at(peer.deadline):await asyncio.gather(*tasks)
            completed=True
        except BaseException:
            self._failed=True
            raise
        finally:
            for sock in (native,ipc):
                if sock is not None:
                    try:sock.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                    sock.close()
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            # Keep successful responder peer alive until the client closes or
            # the finite owner closes it. Immediate close may discard our done
            # packet while it is still in flight over TURN.
            await channel.close(close_peer=not completed)

    async def close(self):
        self._closed=True
        if self._signal is not None:await asyncio.to_thread(self._signal.close)
        if self._task is not None:
            self._task.cancel();await asyncio.gather(self._task,return_exceptions=True)
        if self._peer is not None:await self._peer.close()
