"""RAM-only finite relay. HTTP framing is NOT TLS: production needs a private TLS HF gateway."""
from dataclasses import dataclass, field
import hmac
import math
import os
import re
import threading
import time
import uuid

MAX_CHUNK = 524288
CAPS = {'av': 36 * 1024 * 1024, 'ack': 65536, 'input': 131072}


class RelayError(Exception):
    def __init__(self, status):
        self.status = status


@dataclass(frozen=True)
class RelayConfig:
    session: str
    not_after: float
    producer_key: str = field(repr=False)
    consumer_key: str = field(repr=False)
    av_capacity: int = CAPS['av']
    ack_capacity: int = CAPS['ack']
    input_capacity: int = CAPS['input']
    body_timeout: float = 5.0
    wait_timeout: float = 2.0
    long_session: bool = False

    def __post_init__(self):
        if type(self.session) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', self.session):
            raise ValueError('invalid session')
        if type(self.long_session) is not bool:
            raise ValueError('explicit long session selection required')
        if (type(self.not_after) not in (int,float) or not math.isfinite(self.not_after)
                or self.not_after<=0 or self.not_after-time.time()>(10800 if self.long_session else 1800)):
            raise ValueError('invalid lease')
        if any(type(k) is not str or not re.fullmatch('[0-9a-f]{64}', k) for k in (self.producer_key, self.consumer_key)) or self.producer_key == self.consumer_key:
            raise ValueError('invalid keys')
        for name, maximum in CAPS.items():
            value = getattr(self, name + '_capacity')
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError('invalid capacity')
        if (type(self.body_timeout) not in (int,float) or not math.isfinite(self.body_timeout)
                or type(self.wait_timeout) not in (int,float) or not math.isfinite(self.wait_timeout)
                or not 0 < self.body_timeout <= 5 or not 0 <= self.wait_timeout <= 2):
            raise ValueError('invalid timeout')

    @classmethod
    def from_env(cls):
        return cls(os.environ['H3_RELAY_SESSION'], float(os.environ['H3_RELAY_NOT_AFTER']),
                   os.environ['H3_RELAY_PRODUCER_KEY'], os.environ['H3_RELAY_CONSUMER_KEY'])


class Relay:
    def __init__(self, config):
        self.config = config
        self.boot_id = uuid.uuid4().hex
        self._deadline = time.monotonic() + max(0, config.not_after - time.time())
        self._cv = threading.Condition()
        self._terminal = False
        self._claims = set()
        self._queues = {k: bytearray() for k in CAPS}
        self._reserved = {k: 0 for k in CAPS}
        self._tickets = {}

    def _end(self):
        self._terminal = True
        for q in self._queues.values():
            q.clear()
        self._tickets.clear()
        self._reserved = dict.fromkeys(CAPS, 0)
        self._cv.notify_all()

    def _check(self):
        if time.monotonic() >= self._deadline or time.time() >= self.config.not_after:
            self._end()
        if self._terminal:
            raise RelayError(410)

    def authenticate(self, session, role, key):
        with self._cv:
            self._check()
            if role not in ('producer', 'consumer') or not isinstance(key, str):
                raise RelayError(401)
            expected = getattr(self.config, role + '_key')
            if not hmac.compare_digest(key.encode(), expected.encode()) or session != self.config.session:
                raise RelayError(401)
            return role

    def claimed(self, role):
        with self._cv:
            self._check()
            if role not in self._claims:
                raise RelayError(403)

    def claim(self, role):
        with self._cv:
            self._check()
            if role in self._claims:
                raise RelayError(409)
            self._claims.add(role)

    def reserve(self, channel, size):
        with self._cv:
            self._check()
            if channel not in CAPS or type(size) is not int or not 0 < size <= MAX_CHUNK:
                raise RelayError(413)
            if len(self._tickets)>=8 or len(self._queues[channel]) + self._reserved[channel] + size > getattr(self.config, channel + '_capacity'):
                raise RelayError(429)
            ticket = object()
            self._tickets[ticket] = (channel, size)
            self._reserved[channel] += size
            return ticket

    def commit(self, ticket, body):
        with self._cv:
            self._check()
            channel, size = self._tickets.pop(ticket)
            if len(body) != size:
                self._end()
                raise RelayError(410)
            self._reserved[channel] -= size
            self._queues[channel].extend(body)
            self._cv.notify_all()

    def take(self, channel):
        with self._cv:
            end = min(self._deadline, time.monotonic() + self.config.wait_timeout)
            while True:
                self._check()
                q = self._queues[channel]
                if q:
                    result = bytes(q[:MAX_CHUNK])
                    del q[:len(result)]
                    # Wake capacity waiters immediately: reserved bytes only free
                    # now, so a producer blocked on commit() must not wait a poll.
                    self._cv.notify_all()
                    return result
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return b''
                self._cv.wait(remaining)

    def abort(self):
        with self._cv:
            self._end()

    def status(self):
        with self._cv:
            self._check()
            return dict(schema='h3-live-relay-v1', session=self.config.session,
                        boot_id=self.boot_id, not_after=self.config.not_after,
                        terminal=self._terminal, claimed_roles=sorted(self._claims),
                        queued={k: len(v) for k, v in self._queues.items()})
