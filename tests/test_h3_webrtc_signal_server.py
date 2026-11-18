import importlib.util
import asyncio
import http.client
import json
import socket
import time
import unittest
from types import SimpleNamespace

from research.h3_personal.webrtc_live_peer import Session
from research.h3_personal.webrtc_signal_server import SignalServer


HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class SignalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = 0
        self.gate = asyncio.Event()

        async def accept(value):
            self.calls += 1
            await self.gate.wait()
            return {"type": "answer", "sdp": "answer"}

        self.peer = SimpleNamespace(deadline=time.monotonic() + 60, ice_profile='host-only-v1',
                                    pc=SimpleNamespace(connectionState="new"),
                                    accept_offer=accept)
        self.server = SignalServer(session=self.peer, loop=asyncio.get_running_loop(),
                                   session_id="test", key="k" * 32,
                                   expires_at=time.time() + 40).start()

    async def asyncTearDown(self):
        await asyncio.to_thread(self.server.close)

    def request(self, path, body=None, key=True, method=None):
        connection = http.client.HTTPConnection(*self.server.address, timeout=5)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-H3-RTC-Key"] = "k" * 32
        try:
            connection.request(method or ("POST" if body is not None else "GET"),
                               path, None if body is None else json.dumps(body), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def offer(self):
        return {"session": "test", "boot": self.server.boot,
                "description": {"type": "offer", "sdp": "offer"}}

    async def test_auth_status_health(self):
        status, body = await asyncio.to_thread(self.request, "/health", key=False)
        self.assertEqual((status, body), (200, {"ready": True}))
        self.assertEqual((await asyncio.to_thread(self.request, "/status", key=False))[0], 401)
        status, body = await asyncio.to_thread(self.request, "/status")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"schema", "session", "boot", "expires_at", "peerconnectionstate", "ice_profile"})
        self.assertEqual(len(body["boot"]), 32)
        self.assertEqual(body['ice_profile'],'host-only-v1')
        self.assertEqual((await asyncio.to_thread(self.request, "/media"))[0], 404)

    async def test_bindings_validation_concurrent_replay(self):
        for field, value in (("session", "wrong"), ("boot", "a" * 32)):
            body = self.offer()
            body[field] = value
            self.assertEqual((await asyncio.to_thread(self.request, "/offer", body))[0], 409)
        for bad in ({}, {**self.offer(), "key": "secret"},
                    {**self.offer(), "description": {"type": "offer", "sdp": "x" * 65537}}):
            self.assertEqual((await asyncio.to_thread(self.request, "/offer", bad))[0], 400)
        self.assertEqual(self.calls, 0)
        first = asyncio.create_task(asyncio.to_thread(self.request, "/offer", self.offer()))
        async with asyncio.timeout(3):
            while not self.calls:
                await asyncio.sleep(.01)
        self.assertEqual((await asyncio.to_thread(self.request, "/offer", self.offer()))[0], 409)
        self.gate.set()
        self.assertEqual((await first)[0], 200)
        self.assertEqual((await asyncio.to_thread(self.request, "/offer", self.offer()))[0], 409)
        self.assertEqual(self.calls, 1)

    async def test_error_is_redacted_and_claim_retained(self):
        async def fail(value):
            raise RuntimeError("secret SDP IP key")
        self.peer.accept_offer = fail
        status, body = await asyncio.to_thread(self.request, "/offer", self.offer())
        self.assertEqual((status, body), (502, {"error": "negotiation failed"}))
        self.assertEqual((await asyncio.to_thread(self.request, "/offer", self.offer()))[0], 409)

    async def test_raw_framing(self):
        def raw(headers):
            with socket.create_connection(self.server.address, timeout=3) as sock:
                sock.sendall(b"POST /offer HTTP/1.1\r\nHost: local\r\n" + headers + b"\r\n")
                return sock.recv(4096).split(b"\r\n")[0]
        for headers, code in ((b"Content-Length: 0\r\nContent-Length: 0\r\n", b"400"),
                              (b"Transfer-Encoding: chunked\r\n", b"400"),
                              (b"Expect: 100-continue\r\n", b"400"),
                              (b"Upgrade: websocket\r\n", b"400"),
                              (b"Content-Length: 131073\r\n", b"413")):
            self.assertIn(code, await asyncio.to_thread(raw, headers))
        self.assertEqual(self.calls, 0)

    async def test_real_http_offer_answer_control(self):
        await asyncio.to_thread(self.server.close)
        deadline = time.monotonic() + 25
        async with Session(deadline=deadline, control=True) as caller, Session(deadline=deadline) as receiver:
            self.server = SignalServer(session=receiver, loop=asyncio.get_running_loop(),
                                       session_id="test", key="k" * 32,
                                       expires_at=time.time() + 20).start()
            offer = await caller.create_offer()
            body = self.offer()
            body["description"] = offer
            status, result = await asyncio.to_thread(self.request, "/offer", body)
            self.assertEqual(status, 200)
            await caller.set_answer(result["description"])
            await asyncio.gather(caller.wait_control(), receiver.wait_control())
            await caller.send("hello")
            self.assertEqual(await receiver.receive(), "hello")
            await receiver.send("ack")
            self.assertEqual(await caller.receive(), "ack")
            self.assertEqual((await asyncio.to_thread(self.request, "/offer", body))[0], 409)
            self.assertEqual(receiver.pc.connectionState, "connected")


if __name__ == "__main__":
    unittest.main()
