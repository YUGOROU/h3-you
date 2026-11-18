"""Isolated RTC child, control/SDP via HTTPS; audio only DataChannel/owned IPC."""
import asyncio
from dataclasses import asdict
import http.client
import json
import os
import re
import socket
import sys
import time
import threading
import math

from .asr_rtc_client import request, SignalingFailure, STAGES
from . import paired_av_wire as wire


def report_failure(control,stage):
    if stage not in STAGES:stage='config'
    try:
        wire._write(control,wire._pack(dict(schema='h3-voxcpm-rtc-ipc-error-v1',stage=stage)),time.monotonic()+.2)
    except BaseException:pass


async def serve(config,control,media,progress=lambda stage:None):
    progress('imports')
    from .webrtc_live_peer import Session
    from .webrtc_exact_channel import Binding,ExactChannel
    from .webrtc_exact_ipc import bridge
    from .webrtc_turn import fetch_turn_credentials
    from .webrtc_ice import HF_CLOUDFLARE_TURN_TLS
    deadline=config['deadline'];token=os.environ.pop('H3_VOX_RTC_TOKEN');key=os.environ.pop('H3_VOX_RTC_KEY')
    async with asyncio.timeout_at(deadline):
        progress('credentials')
        turn=await asyncio.to_thread(fetch_turn_credentials,token,600)
        progress('start')
        row=await asyncio.to_thread(request,config['endpoint'],'/rtc/start',token,key,config['session'],deadline,
            dict(session=config['session'],boot=config['boot'],epoch=config['epoch'],text=config['text'],turn_config=turn))
        if (type(row) is not dict or set(row)!={'schema','binding','signal_boot','expires_at','ice_profile'}
                or row['schema']!='h3-voxcpm-rtc-turn-v1' or row['ice_profile']!=HF_CLOUDFLARE_TURN_TLS
                or type(row['expires_at']) not in (int,float) or not time.time()<row['expires_at']<=config['expires_at']):
            raise ValueError('ASR turn identity')
        progress('binding')
        binding=Binding(**row['binding'])
        if (binding.session!=config['session'] or binding.epoch!=config['epoch'] or binding.purpose!='voxcpm-native-wire-v1'
                or binding.max_bytes!=8*1024*1024 or binding.boot!=config['boot'] or type(row['signal_boot']) is not str or not re.fullmatch('[a-f0-9]{32}',row['signal_boot'])):
            raise ValueError('ASR binding mismatch')
        deadline=min(deadline,time.monotonic()+row['expires_at']-time.time())
        async with Session(deadline=deadline,control=True,ice_profile=HF_CLOUDFLARE_TURN_TLS,turn_config=turn) as peer:
            progress('offer')
            offer=await peer.create_offer()
            progress('answer')
            answer=await asyncio.to_thread(request,config['endpoint'],'/rtc/offer',token,key,config['session'],deadline,
                dict(session=binding.session,boot=row['signal_boot'],description=offer))
            if type(answer) is not dict or set(answer)!={'session','boot','description'} or answer['session']!=binding.session or answer['boot']!=row['signal_boot']:
                raise ValueError('ASR answer identity')
            progress('control')
            await peer.set_answer(answer['description']);await peer.wait_control()
            progress('channel')
            channel=ExactChannel(peer,binding);await channel.start()
            progress('ready')
            await asyncio.to_thread(wire._write,control,wire._pack(dict(schema='h3-voxcpm-rtc-ipc-ready-v1',binding=asdict(binding))),deadline)
            progress('bridge')
            result=await bridge(channel,media,progress)
            result.update(received_bytes=channel._received,received_sha256=channel._rx.hexdigest())
            progress('complete')
            await asyncio.to_thread(wire._write,control,wire._pack(dict(schema='h3-voxcpm-rtc-ipc-complete-v1',**result)),deadline)


def main():
    control=socket.socket(fileno=int(sys.argv[1]));media=socket.socket(fileno=int(sys.argv[2]))
    state=['config'];timer=None
    def progress(stage):state[0]=stage
    def expired():
        # asyncio.run waits for executor threads; a process-local timer does not.
        report_failure(control,state[0])
        os._exit(1)
    try:
        config=wire._json(control,time.monotonic()+10)
        deadline=config['deadline'];remaining=deadline-time.monotonic()
        if type(deadline) not in (int,float) or not math.isfinite(deadline) or not 0<remaining<=120:
            raise ValueError('finite child deadline required')
        timer=threading.Timer(remaining,expired);timer.daemon=True;timer.start()
        asyncio.run(serve(config,control,media,progress));return 0
    except BaseException as error:
        report_failure(control,error.stage if isinstance(error,SignalingFailure) else state[0]);return 1
    finally:
        if timer is not None:timer.cancel()
        control.close();media.close()

if __name__=='__main__':raise SystemExit(main())
