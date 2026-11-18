"""Explicit ICE profiles. No environment activation or unapproved relays."""
HOST_ONLY = 'host-only-v1'
CLOUDFLARE_STUN = 'cloudflare-stun-v1'
STUN_URL = 'stun:stun.cloudflare.com:3478'
HF_CLOUDFLARE_TURN_TLS = 'hf-cloudflare-turn-tls-v1'
TURN_TLS_URL = 'turns:turn.cloudflare.com:443?transport=tcp'


def ice_urls(profile=HOST_ONLY):
    if type(profile) is not str:
        raise ValueError('explicit ICE profile required')
    if profile == HOST_ONLY:
        return []
    if profile == CLOUDFLARE_STUN:
        return [STUN_URL]
    if profile == HF_CLOUDFLARE_TURN_TLS:
        return [TURN_TLS_URL]
    raise ValueError('unapproved ICE profile')
