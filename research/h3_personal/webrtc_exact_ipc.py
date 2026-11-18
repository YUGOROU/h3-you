"""Owned local socketpair↔exact WebRTC channel, no TCP/HTTP media path."""
import asyncio
import errno
import socket
from .webrtc_exact_channel import MAX_CHUNK

async def bridge(channel,sock,progress=lambda stage:None,*,keep_peer=False):
    if not isinstance(sock,socket.socket) or sock.family!=socket.AF_UNIX or sock.type!=socket.SOCK_STREAM:
        raise ValueError('owned local socketpair required')
    sock.getpeername();sock.setblocking(False)
    loop=asyncio.get_running_loop();local_eof_already_closed=False
    async def outgoing():
        while True:
            async with asyncio.timeout_at(channel.peer.deadline):
                body=await loop.sock_recv(sock,MAX_CHUNK)
            if not body:return await channel.finish()
            await channel.send(body)
    async def incoming():
        nonlocal local_eof_already_closed
        async def sink(body):
            async with asyncio.timeout_at(channel.peer.deadline):await loop.sock_sendall(sock,body)
        while await channel.receive_into(sink):pass
        try:sock.shutdown(socket.SHUT_WR)
        except OSError as error:
            # No data write is suppressed: carrier EOF was independently
            # verified after every byte was handed to the local socket. ASR
            # may already have closed after its own native final coverage.
            if error.errno!=errno.ENOTCONN:raise
            local_eof_already_closed=True
    tasks=[asyncio.create_task(outgoing()),asyncio.create_task(incoming())]
    completed=False
    try:
        async with asyncio.timeout_at(channel.peer.deadline):
            result=await asyncio.gather(*tasks)
            progress('bridge-barrier')
            await channel.barrier()
        completed=True
        return dict(result[0],local_eof_already_closed=local_eof_already_closed)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        try:sock.shutdown(socket.SHUT_RDWR)
        except OSError:pass
        sock.close();await channel.close(close_peer=not (keep_peer and completed))
