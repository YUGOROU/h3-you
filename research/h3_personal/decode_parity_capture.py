"""Explicit diagnostic-only in-memory committed-suffix capture; no tensor export.

The normal decoder calls this only for explicit manifest parallel acceptance or
H3_SMD_DECODE_PARITY. Each rank compares its own identical cumulative input;
finite process-group/provider deadlines remain the caller's responsibility.
"""
import hashlib
import json
import os
from pathlib import Path

from .continuation_plan import video_t
from .decode_parity import compare_decode

ENV = 'H3_SMD_DECODE_PARITY'


def _digest(value):
    descriptor = json.dumps([list(value.shape), 'float32'], separators=(',', ':')).encode()
    return hashlib.sha256(descriptor + b'\0' + value.numpy().tobytes()).hexdigest()


class Capture:
    def __init__(self, manifest):
        self.parallel = manifest.get('parallel_acceptance') is True
        self.session = manifest['session_id']
        self.total = manifest['total_frames']
        if (not isinstance(self.session, str) or not self.session or
                type(self.total) is not int or self.total not in (124, 226, 328, 430)):
            raise ValueError('bounded continuation session required')
        self.parts = []
        self.arrivals = []
        self.index = 0
        self.geometry = None
        self.closed = self.failed = False

    def fault(self):
        self.failed = self.closed = True
        self.parts.clear()

    def append(self, manifest, normalized):
        import torch
        if (self.closed or self.failed or manifest['session_id'] != self.session or
                manifest['total_frames'] != self.total or
                type(manifest['index']) is not int or manifest['index'] != self.index):
            raise ValueError('stale, closed or foreign parity capture')
        final = self.index == (self.total-124)//102
        if type(manifest['final']) is not bool or manifest['final'] != final:
            raise ValueError('parity capture final grid differs')
        shape = tuple(normalized.shape)
        count = 37 if self.index == 0 else 30
        if (len(shape) != 5 or shape[:3] != (1, 24, count) or
                any(type(n) is not int or n <= 0 or n % 2 for n in shape[3:]) or
                shape[3]*shape[4] > 56*32):
            raise ValueError('bounded normalized suffix NCTHW required')
        if self.geometry is not None and shape[3:] != self.geometry:
            raise ValueError('parity capture canvas drift')
        part = normalized.detach().to(device='cpu', dtype=torch.float32).clone().contiguous()
        if not bool(torch.isfinite(part).all().item()):
            raise ValueError('finite committed normalized suffix required')
        self.geometry = shape[3:]
        self.parts.append(part)
        self.arrivals.append(count)
        self.index += 1

    def finish(self, vae):
        import torch
        try:
            error = None
            try:
                if (self.failed or self.closed or self.index != (self.total-124)//102+1 or
                        sum(self.arrivals) != video_t(self.total)):
                    raise ValueError('complete committed timeline required')
                normalized = torch.cat(self.parts, dim=2)
                digest = _digest(normalized)
                device_input = normalized.to(device=vae.latents_mean.device)
            except Exception as exc:
                error = exc
            if self.parallel:
                d = torch.distributed
                if not d.is_initialized() or d.get_world_size() != 2:
                    raise ValueError('parallel parity requires initialized two-rank group')
                ok = torch.tensor(int(error is None), dtype=torch.int32, device=vae.latents_mean.device)
                d.all_reduce(ok, op=d.ReduceOp.MIN)
                if not bool(ok.item()):
                    raise RuntimeError('parallel capture preparation rejected') from error
            if error is not None:
                raise error
            comparator = compare_decode
            if self.parallel:
                from .parallel_decode_parity import compare_decode as comparator
            result = comparator(vae, device_input,
                                    arrivals=tuple(self.arrivals), atol=0.0)
            if _digest(normalized) != digest:
                raise ValueError('captured CPU timeline mutated')
            result.update(session_id=self.session, total_frames=self.total,
                          input_sha256=digest)
            self.closed = True
            return result
        except BaseException:
            self.fault()
            raise
        finally:
            self.parts.clear()


def _agreement(stage, check):
    """Local capture/compare/write errors join the same fixed MIN on both ranks."""
    import torch
    d = torch.distributed
    capture = getattr(stage, '_h3_decode_parity_capture', None)
    if not d.is_initialized() or d.get_world_size() != 2:
        if capture is not None:
            capture.fault()
        raise ValueError('parity diagnostic requires initialized two-rank group')
    result, error = None, None
    try:
        result = check()
    except Exception as exc:
        error = exc
    ok = torch.tensor(0 if error else 1, dtype=torch.int32, device=stage.vae.latents_mean.device)
    try:
        d.all_reduce(ok, op=d.ReduceOp.MIN)
        if not bool(ok.item()):
            raise RuntimeError('collective parity diagnostic rejected') from error
        return result
    except BaseException:
        capture = getattr(stage, '_h3_decode_parity_capture', None)
        if capture is not None:
            capture.fault()
        raise


def record_suffix(stage, context, normalized):
    def record():
        manifest = context['manifest']
        capture = getattr(stage, '_h3_decode_parity_capture', None)
        if capture is None:
            if manifest['index'] != 0:
                raise ValueError('capture must start at window zero')
            capture = Capture(manifest)
            stage._h3_decode_parity_capture = capture
        capture.append(manifest, normalized)
        return capture
    return _agreement(stage, record)


def finish_parity(stage, capture):
    def finish():
        import torch
        result = capture.finish(stage.vae)
        rank = torch.distributed.get_rank()
        result['worker_rank'] = rank
        folder = Path(os.environ['H3_REF2VA_COVERAGE_DIR'])
        prefix = 'parallel-parity' if capture.parallel else 'decode-parity'
        path = folder / f'{prefix}-rank-{rank}.json'
        if path.exists():
            raise ValueError('never overwrite a parity diagnostic')
        temporary = path.with_suffix('.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(result, stream, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
        return result
    return _agreement(stage, finish)
