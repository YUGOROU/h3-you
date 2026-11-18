"""Isolated RTC process for native ASR Endpoint. No model dependencies here."""
import asyncio
import http.client
import json
import os
import signal
import threading
import time


def native_ready():
    connection=http.client.HTTPConnection('127.0.0.1',7861,timeout=3)
    try:
        connection.request('GET','/ready');response=connection.getresponse();body=response.read(16385)
        if response.status!=200 or len(body)>16384:raise ValueError('native ASR not ready')
        row=json.loads(body)
        if type(row) is not dict or row.get('device')!='cuda':raise ValueError('native CUDA required')
        return row
    finally:connection.close()


async def serve(key,expires_at):
    from research.h3_personal.asr_rtc_service import ASRRTCManager
    from research.h3_personal.asr_rtc_gateway import Server
    async def ready():return await asyncio.to_thread(native_ready)
    manager=ASRRTCManager(key=key,expires_at=expires_at,native_ready=ready)
    server=Server(('0.0.0.0',7860),manager,asyncio.get_running_loop())
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True)
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop.set)
    thread.start()
    try:
        async with asyncio.timeout(max(.001,expires_at-time.time())):await stop.wait()
    except TimeoutError:pass
    finally:
        await manager.close();await asyncio.to_thread(server.shutdown);server.server_close();thread.join(2)
        if thread.is_alive():raise RuntimeError('ASR control server cleanup unconfirmed')


def main():
    key=os.environ.pop('H3_ASR_KEY');expires=int(os.environ['H3_ASR_EXPIRES_AT'])
    if os.environ.get('H3_ASR_MEDIA_TRANSPORT')!='webrtc-exact-v1':raise ValueError('explicit RTC ASR deployment required')
    if not 0<expires-time.time()<=1800:raise ValueError('finite ASR process lease')
    asyncio.run(serve(key,expires))

if __name__=='__main__':
    try:main()
    except BaseException:raise SystemExit(1)  # No private diagnostics in service logs.
