"""Isolated RTC gateway; native model/media interface is private loopback only."""
import asyncio
import http.client
import json
import os
import signal
import time

async def main():
    from research.h3_personal.voxcpm_private_gate import PrivateGate
    from research.h3_personal.voxcpm_rtc_service import VoxRTCManager,Server
    if os.environ.get('H3_VOX_MEDIA_TRANSPORT')!='webrtc-exact-v1':raise ValueError('explicit RTC runtime required')
    expiry=int(os.environ['H3_TTS_EXPIRES_AT'])
    if not 0<expiry-time.time()<=1200:raise ValueError('finite native lease required')
    def ready():
        c=http.client.HTTPConnection('127.0.0.1',7861,timeout=1)
        try:
            c.request('GET','/health');r=c.getresponse()
            return r.status==200 and json.loads(r.read(4096)).get('ready') is True
        except (OSError,ValueError):return False
        finally:c.close()
    while time.time()<expiry-15:
        if await asyncio.to_thread(ready):break
        await asyncio.sleep(.2)
    else:raise TimeoutError('native model readiness expired')
    gate=PrivateGate(key=os.environ.pop('H3_TTS_KEY'),session=os.environ['H3_TTS_SESSION'],
        boot=os.environ['H3_TTS_BOOT'],expires_at=expiry)
    await asyncio.to_thread(gate.ready,0)
    manager=VoxRTCManager(gate);server=Server(('0.0.0.0',7860),manager,asyncio.get_running_loop())
    import threading
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1});thread.start()
    stopped=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stopped.set)
    try:
        async with asyncio.timeout(max(.001,expiry-time.time())):await stopped.wait()
    except TimeoutError:pass
    finally:
        await manager.close();await asyncio.to_thread(server.shutdown);server.server_close();thread.join(2)

if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException:raise SystemExit(1)
