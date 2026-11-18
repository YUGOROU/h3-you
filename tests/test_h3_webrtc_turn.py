"""Offline broker security/normalization tests; only synthetic secrets."""
import copy
import json
import traceback
import unittest
from unittest.mock import Mock, patch

from research.h3_personal import webrtc_turn as turn
from research.h3_personal.webrtc_ice import TURN_TLS_URL


def broker():
    return {'iceServers': [
        {'urls': sorted(turn._STUN)},
        {'urls': sorted(turn._TURN), 'username': 'test-user', 'credential': 'test-pass'},
    ]}


def config():
    return {'urls': TURN_TLS_URL, 'username': 'test-user', 'credential': 'test-pass'}


class TurnTests(unittest.TestCase):
    def fetch(self, body=None, status=200, error=None, ttl=600):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = status
        response.read.return_value = json.dumps(broker()).encode() if body is None else body
        if error:
            connection.request.side_effect = error
        with patch.object(turn.http.client, 'HTTPSConnection', return_value=connection) as factory:
            try:
                result = turn.fetch_turn_credentials('synthetic-hf-token', ttl)
                return result, connection, factory
            finally:
                connection.close.assert_called_once_with()

    def test_fetch_fixed_destination_header_only(self):
        result, connection, factory = self.fetch(ttl=1)
        self.assertEqual(result, config())
        factory.assert_called_once_with('fastrtc-turn-service.hf.space', timeout=20)
        connection.request.assert_called_once_with(
            'GET', '/credentials?ttl=1',
            headers={'Authorization': 'Bearer synthetic-hf-token'})
        connection.getresponse.return_value.read.assert_called_once_with(16385)

    def test_normalize_and_copy(self):
        source = broker()
        self.assertEqual(turn.extract_turn_credentials(source), config())
        value = config()
        self.assertIsNot(turn.validate_turn_config(value), value)
        self.assertEqual(source, broker())

    def test_config_rejections(self):
        for key, bad in [('urls', [TURN_TLS_URL]), ('urls', 'turns:evil:443'),
                         ('username', ''), ('username', 'a' * 1025),
                         ('credential', 'a' * 4097), ('credential', 123)]:
            value = config()
            value[key] = bad
            with self.subTest(key=key, kind=type(bad)):
                with self.assertRaises(ValueError):
                    turn.validate_turn_config(value)
        for bad in ['x y', 'x\ny', 'x\ry', 'x\ty', '\x00', '\x7f', 'é']:
            for key in ('username', 'credential'):
                value = config()
                value[key] = bad
                with self.assertRaises(ValueError):
                    turn.validate_turn_config(value)
        for value in [None, [], {}, dict(config(), secret='hidden')]:
            with self.assertRaises(ValueError):
                turn.validate_turn_config(value)
        self.assertEqual(turn.validate_turn_config(dict(config(), username='a' * 1024,
                                                       credential='b' * 4096))['username'], 'a' * 1024)

    def test_schema_rejections(self):
        cases = [None, [], {}, {'iceServers': []}, {'iceServers': [None]},
                 {'iceServers': broker()['iceServers'] * 5},
                 dict(broker(), secret='hidden')]
        for change in [
            {'urls': [TURN_TLS_URL.replace('cloudflare.com', 'evil.example')]},
            {'urls': ['turn:turn.cloudflare.com:443?transport=udp']},
            {'urls': TURN_TLS_URL}, {'urls': []}, {'urls': [TURN_TLS_URL] * 9},
            {'secret': 'hidden'}, {'credential': False},
        ]:
            value = copy.deepcopy(broker())
            value['iceServers'][1].update(change)
            cases.append(value)
        cases.extend([
            {'iceServers': [{'urls': [TURN_TLS_URL]}]},
            {'iceServers': [broker()['iceServers'][1]] * 2},
        ])
        for value in cases:
            with self.subTest(index=cases.index(value)):
                with self.assertRaises(ValueError):
                    turn.extract_turn_credentials(value)

    def test_status_no_redirect_or_retry(self):
        for status in (201, 301, 302, 307, 308, 401, 403, 500):
            connection = Mock()
            connection.getresponse.return_value.status = status
            with patch.object(turn.http.client, 'HTTPSConnection', return_value=connection) as factory:
                with self.assertRaises(ValueError):
                    turn.fetch_turn_credentials('synthetic-hf-token')
                factory.assert_called_once()
                connection.request.assert_called_once()
                connection.getresponse.return_value.read.assert_not_called()
                connection.close.assert_called_once()

    def test_body_rejections(self):
        for body in [b'x' * 16385, b'not-json', b'\xff', b'[]',
                     b'{"iceServers":[],"iceServers":[]}',
                     b'{"iceServers":[{"urls":[],"urls":[]}]}']:
            with self.assertRaises(ValueError):
                self.fetch(body=body)

    def test_validation_before_network(self):
        with patch.object(turn.http.client, 'HTTPSConnection') as factory:
            for ttl in (True, False, 0, 601, 1.0, '600', None):
                with self.assertRaises(ValueError):
                    turn.fetch_turn_credentials('synthetic', ttl)
            for token in (None, '', 'x\r\ny', 'x y', 123):
                with self.assertRaises(ValueError):
                    turn.fetch_turn_credentials(token)
            factory.assert_not_called()

    def test_errors_sanitized(self):
        try:
            self.fetch(error=OSError('synthetic-hf-token test-pass hidden-body'))
        except ValueError as exc:
            rendered = ''.join(traceback.format_exception_only(type(exc), exc))
            for secret in ('synthetic-hf-token', 'test-pass', 'hidden-body'):
                self.assertNotIn(secret, str(exc))
                self.assertNotIn(secret, rendered)
            self.assertIsNone(exc.__cause__)
            self.assertTrue(exc.__suppress_context__)
        else:
            self.fail('expected sanitized failure')


if __name__ == '__main__':
    unittest.main()
