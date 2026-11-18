"""Transport/CLI boundary checks: fake sockets only, never a live OBS server."""
import base64
import hashlib
import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from research.h3_personal.obs_socket import LocalOBSWebSocket
from research.run_h3_obs_diagnostics import main


def server_frame(data, opcode=1, final=True):
    data = data.encode() if isinstance(data, str) else data
    length = bytes([len(data)]) if len(data) < 126 else b'\x7e'+struct.pack('!H', len(data))
    return bytes([(128 if final else 0)|opcode])+length+data


class FakeSocket:
    def __init__(self, clock):
        self.clock = clock
        self.buffer = bytearray()
        self.sent = []
        self.closed = False
        self.advance = 0
        self.bad_upgrade = False

    def settimeout(self, seconds):
        self.timeout = seconds

    def sendall(self, data):
        self.sent.append(data)
        if data.startswith(b'GET'):
            key = next(line.split(b': ', 1)[1] for line in data.split(b'\r\n')
                       if line.startswith(b'Sec-WebSocket-Key:'))
            accept = base64.b64encode(hashlib.sha1(key+b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest())
            self.buffer.extend(b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n'
                               b'Connection: keep-alive, Upgrade\r\nSec-WebSocket-Accept: '+
                               (b'bad' if self.bad_upgrade else accept)+b'\r\n\r\n')

    def recv(self, count):
        self.clock[0] += self.advance
        data = bytes(self.buffer[:count]);del self.buffer[:count]
        return data

    def close(self):
        self.closed = True


class OBSSocketTests(unittest.TestCase):
    def setUp(self):
        self.clock = [0]
        self.socket = FakeSocket(self.clock)
        self.connector = patch('research.h3_personal.obs_socket.socket.create_connection', return_value=self.socket).start()
        self.addCleanup(patch.stopall)

    def connect(self, **kwargs):
        return LocalOBSWebSocket(clock=lambda: self.clock[0], **kwargs)

    def test_local_only_upgrade_and_masked_client_text(self):
        transport = self.connect()
        self.connector.assert_called_once_with(('127.0.0.1', 4455), timeout=5)
        transport.send('{"op":1}')
        self.assertEqual(self.socket.sent[-1][0], 0x81)
        self.assertTrue(self.socket.sent[-1][1] & 128)
        self.socket.buffer.extend(server_frame('hello'))
        self.assertEqual(transport.recv(timeout=1), 'hello')
        transport.close();transport.close()
        self.assertTrue(self.socket.closed)

    def test_fragmented_text_and_ping_do_not_reset_deadline(self):
        transport = self.connect()
        self.socket.buffer.extend(server_frame('hel', final=False)+
                                  server_frame(b'ping', opcode=9)+server_frame('lo', opcode=0))
        self.assertEqual(transport.recv(timeout=1), 'hello')
        self.assertEqual(self.socket.sent[-1][0], 0x8a)

    def test_bad_upgrade_closes_socket(self):
        self.socket.bad_upgrade = True
        with self.assertRaises(ValueError):self.connect()
        self.assertTrue(self.socket.closed)

    def test_partial_reads_cannot_extend_receive_deadline(self):
        transport = self.connect(timeout=3)
        self.socket.buffer.extend(server_frame('a'*40))
        self.socket.advance = .3
        with self.assertRaises(TimeoutError):transport.recv(timeout=.5)
        self.assertLess(self.clock[0], 1)
        transport.close()

    def test_invalid_frames_fail_closed(self):
        for payload in (server_frame(b'abc', opcode=2), b'\x81\x80',
                        b'\x81\x7f'+struct.pack('!Q', 65537),
                        server_frame(b'ping', opcode=9, final=False)):
            self.socket = FakeSocket(self.clock)
            self.connector.return_value = self.socket
            transport = self.connect()
            self.socket.buffer.extend(payload)
            with self.assertRaises((ValueError, EOFError)):transport.recv(timeout=1)
            transport.close()

    def test_offline_cli_has_no_socket_or_environment_dependency(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertFalse(json.loads(output.getvalue())['network'])
        self.connector.assert_not_called()

    def test_cli_connection_errors_never_echo_private_exception(self):
        self.connector.side_effect = OSError('private-secret-marker')
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--run']), 1)
        self.assertNotIn('private-secret-marker', output.getvalue())
        self.assertIn('localhost_connection_or_upgrade_failed', output.getvalue())


if __name__ == '__main__':unittest.main()
