"""Separate free/silent capture attempts from bounded, charged Pi requests.

Limits never extend ASR/H3 deadlines or allow additional Pi reservations.
Call these helpers under the resident owner's lock.
"""
MAX_PI_REQUESTS = 4
MAX_CAPTURE_ATTEMPTS = 12


def admit_capture(attempts, pi_requests):
    if type(attempts) is not int or type(pi_requests) is not int or min(attempts, pi_requests) < 0:
        raise ValueError('nonnegative integer counters required')
    if attempts >= MAX_CAPTURE_ATTEMPTS:
        raise ValueError('capture attempt limit')
    if pi_requests >= MAX_PI_REQUESTS:
        raise ValueError('Pi request limit')


def reserve_pi_request(state):
    """Count BEFORE sending, even if acknowledgement is later lost. Never refund."""
    count = state['pi_requests']
    if type(count) is not int or not 0 <= count < MAX_PI_REQUESTS:
        raise ValueError('Pi request limit')
    state['pi_requests'] = count + 1
    return state['pi_requests']
