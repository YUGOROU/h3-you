"""Memory-only HF TURN broker client; no network or credential discovery on import.

Broker contract: FastRTC credentials.py at
59ca4f77e1669a9599c0895fd0d7c39f356bf89f. Only TLS/TCP 443 is returned.
Callers own credential lifetime and must not log or persist returned dictionaries.
"""
import http.client
import json

from .webrtc_ice import TURN_TLS_URL

_BROKER_HOST = 'fastrtc-turn-service.hf.space'
_MAX_BODY = 16 * 1024
_STUN = frozenset({'stun:stun.cloudflare.com:3478', 'stun:stun.cloudflare.com:443'})
_TURN = frozenset({
    'turn:turn.cloudflare.com:3478?transport=udp',
    'turn:turn.cloudflare.com:3478?transport=tcp',
    'turns:turn.cloudflare.com:5349?transport=tcp',
    'turn:turn.cloudflare.com:443?transport=udp',
    'turn:turn.cloudflare.com:80?transport=tcp',
    TURN_TLS_URL,
})


def _secret(value, limit):
    return (type(value) is str and 1 <= len(value) <= limit
            and all(33 <= ord(char) <= 126 for char in value))


def validate_turn_config(value):
    """Validate and copy the exact minimal TLS-only config; raise ValueError."""
    if (type(value) is not dict or set(value) != {'urls', 'username', 'credential'}
            or type(value['urls']) is not str or value['urls'] != TURN_TLS_URL
            or not _secret(value['username'], 1024)
            or not _secret(value['credential'], 4096)):
        raise ValueError('invalid TURN configuration')
    return dict(value)


def extract_turn_credentials(value):
    """Select TLS443 from the bounded official broker schema, rejecting extras."""
    if type(value) is not dict or set(value) != {'iceServers'}:
        raise ValueError('invalid TURN broker schema')
    rows = value['iceServers']
    if type(rows) is not list or not 1 <= len(rows) <= 8:
        raise ValueError('invalid TURN broker schema')
    selected = None
    for row in rows:
        if type(row) is not dict or set(row) not in (
                {'urls'}, {'urls', 'username', 'credential'}):
            raise ValueError('invalid TURN broker schema')
        urls = row['urls']
        if (type(urls) is not list or not 1 <= len(urls) <= 8
                or any(type(url) is not str or url not in _STUN | _TURN for url in urls)
                or len(set(urls)) != len(urls)):
            raise ValueError('invalid TURN broker URLs')
        if set(row) == {'urls'}:
            if any(url not in _STUN for url in urls):
                raise ValueError('missing TURN broker credentials')
        else:
            if (any(url not in _TURN for url in urls)
                    or not _secret(row['username'], 1024)
                    or not _secret(row['credential'], 4096)):
                raise ValueError('invalid TURN broker credentials')
            if TURN_TLS_URL in urls:
                if selected is not None:
                    raise ValueError('ambiguous TURN broker credentials')
                selected = validate_turn_config({
                    'urls': TURN_TLS_URL, 'username': row['username'],
                    'credential': row['credential'],
                })
    if selected is None:
        raise ValueError('missing TLS443 TURN credentials')
    return selected


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate TURN broker key')
        result[key] = value
    return result


def fetch_turn_credentials(token, ttl=600):
    """Fetch once over fixed HTTPS with a 20s socket timeout, no redirects/retry.

    ttl is an exact int in 1..600. Transport, HTTP and parsing failures become
    a generic ValueError without response bodies, credentials or exception chains.
    """
    if type(ttl) is not int or not 1 <= ttl <= 600:
        raise ValueError('invalid TURN credential TTL')
    if not _secret(token, 4096):
        raise ValueError('invalid HF token')
    connection = None
    try:
        connection = http.client.HTTPSConnection(_BROKER_HOST, timeout=20)
        connection.request('GET', '/credentials?ttl=' + str(ttl), headers={
            'Authorization': 'Bearer ' + token,
        })
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('TURN broker HTTP failure')
        body = response.read(_MAX_BODY + 1)
        if len(body) > _MAX_BODY:
            raise ValueError('TURN broker response too large')
        value = json.loads(body.decode('utf-8'), object_pairs_hook=_unique_object)
        return extract_turn_credentials(value)
    except Exception:
        raise ValueError('TURN broker request failed') from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
