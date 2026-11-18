"""Strict, experiment-only idle anchor policy parsing (no environment access)."""

ENVIRONMENT_VARIABLE = 'H3_IDLE_ANCHOR_DIAGNOSTIC'
DIAGNOSTIC_POLICY = 'first-listening-tail-every3-v1'


def parse_idle_anchor_policy(value):
    """Return the explicit diagnostic interval, or None when absent.

    Empty strings, coercions and alternative spellings are deliberately rejected.
    """
    if value is None:
        return None
    if type(value) is str and value == DIAGNOSTIC_POLICY:
        return 3
    raise ValueError('invalid H3_IDLE_ANCHOR_DIAGNOSTIC policy')
