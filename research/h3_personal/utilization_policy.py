"""Strict JSON-only immutable request contracts. No device imports."""
TRACE = dict(schema='h3-worker-trace-v1', selected_window=1,
             max_events=200000, max_bytes=262144)
DENOISE = dict(schema='h3-local-post-compile-v1', max_backend_compiles=8,
               max_compile_seconds=180, max_regions=50)


def normalize(value, kind):
    if value is None:
        return None
    expected = {'trace': TRACE, 'denoise': DENOISE}[kind]
    if (type(value) is not dict or set(value) != set(expected) or
            any(type(value[k]) is not type(v) or value[k] != v
                for k, v in expected.items())):
        raise ValueError('invalid utilization policy')
    return dict(expected)


def policies(trace_policy=None, denoise_policy=None):
    return normalize(trace_policy, 'trace'), normalize(denoise_policy, 'denoise')


def validate_mode(trace, denoise, *, continuation, conflicting=False):
    if (trace is not None or denoise is not None) and (not continuation or conflicting):
        raise ValueError('utilization requires isolated continuation owner')
