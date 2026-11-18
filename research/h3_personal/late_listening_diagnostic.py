"""Isolated scalar-only late-listening diagnostic; disabled unless explicitly bound."""
import json
import os
import stat
from pathlib import Path

ENV = 'H3_REF2VA_LATE_DIAGNOSTIC_FILE'
SCHEMA = 'h3-late-listening-diagnostic-v1'


def read_config():
    path = os.environ.get(ENV)
    if path is None:
        return None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise ValueError('bounded regular diagnostic JSON required')
        raw = os.read(fd, 4097)
        if len(raw) > 4096:
            raise ValueError('diagnostic JSON too large')
    finally:
        os.close(fd)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate diagnostic key')
            result[key] = value
        return result
    c = json.loads(raw, object_pairs_hook=pairs)
    keys = {'schema', 'phase', 'position_policy', 'conversation_id', 'expected_frames', 'max_windows'}
    if type(c) is not dict or set(c) != keys or type(c['phase']) is not int or c['phase'] not in (0, 1):
        raise ValueError('exact diagnostic config required')
    p = c['phase']
    if (c['schema'] != SCHEMA or c['position_policy'] != ('global-v1', 'window-local-v1')[p]
            or c['conversation_id'] != ('late-listening-global', 'late-listening-local')[p]
            or type(c['expected_frames']) is not int or c['expected_frames'] != 634
            or type(c['max_windows']) is not int or c['max_windows'] != 8):
        raise ValueError('diagnostic binding drift')
    return c


def _binding(batch, c):
    from .streaming_native_stages import validate_manifest
    context = batch.extra['_h3_continuation']
    m = validate_manifest(context['manifest'])
    if m['id'] != c['conversation_id'] or m['max_windows'] != c['max_windows']:
        raise ValueError('diagnostic requires bound active streaming context')
    return context, m


def position_policy(batch):
    c = read_config()
    if c is None:
        return 'global-v1'
    _binding(batch, c)
    return c['position_policy']


def _save(c, rank, receipt):
    directory = Path(os.environ['H3_REF2VA_COVERAGE_DIR'])
    if not directory.is_dir():raise ValueError('existing private coverage directory required')
    if directory.is_symlink():
        raise ValueError('symlink coverage directory')
    path = directory / f"late-listening-{c['phase']}-rank-{rank}.json"
    payload = json.dumps(receipt, allow_nan=False, sort_keys=True).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as out:
            out.write(payload)
            out.flush()
        os.fchmod(fd, 0o400)
    finally:
        os.close(fd)


def capture(stage, batch, context, z, output, final, ops):
    c = read_config()
    if c is None:
        return
    actual, m = _binding(batch, c)
    if actual is not context or context['conversation_owner'].conversation_id != c['conversation_id']:
        raise ValueError('diagnostic owner drift')
    state = getattr(stage, '_h3_late_listening_diagnostic', None)
    if state is None or state['final']:
        if state is not None and state['config'] == c:
            raise ValueError('diagnostic phase already completed')
        state = dict(config=c, final=False, windows=[], latents=[], pixels=[], received=0, frames=0)
        stage._h3_late_listening_diagnostic = state
    if state['config'] != c or m['sequence'] != len(state['windows']):
        raise ValueError('diagnostic sequence/config drift')
    count = int(z.shape[2])
    frames = int(output.shape[2]) if ops.rank == 0 else 0
    if len(state['windows']) >= 8 or state['received'] + count > 187 or state['frames'] + frames > 634:
        raise ValueError('diagnostic retention bound exceeded')
    row = dict(sequence=m['sequence'], role=m['role'], latent_count=count)
    if ops.rank == 0:
        import torch
        cpu = z.detach().to(device='cpu').clone()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError('nonfinite diagnostic latents')
        layout=batch.extra['minimax_h3_layout']
        vt=layout.position_ids[layout.video_indices[layout.num_condition_video_rows:],0]
        at=layout.position_ids[layout.audio_indices[layout.num_condition_audio_rows:],0]
        row.update(mean=float(cpu.float().mean()), std=float(cpu.float().std(unbiased=False)),
                   max=float(cpu.max()),global_temporal_offset=float(context['token'].temporal_offset),
                   model_video_time_min=float(vt.min()),model_video_time_max=float(vt.max()),
                   model_audio_time_min=float(at.min()),model_audio_time_max=float(at.max()))
        if output.dtype != torch.uint8:
            raise ValueError('uint8 incremental output required')
        state['latents'].append(cpu)
        state['pixels'].append(output.detach().cpu().clone())
    state['received'] += count
    state['frames'] += frames
    state['windows'].append(row)
    if not final:
        return
    if state['received'] != 187 or context['conversation_owner'].committedF != 634:
        raise ValueError('diagnostic final timeline mismatch')
    receipt = dict(schema=SCHEMA, **{k: v for k, v in c.items() if k != 'schema'},
                   rank=ops.rank, completed=True, latent_count=state['received'],
                   windows=state['windows'], native_reference=ops.rank == 0,
                   committed_frames=634)
    if ops.rank == 0:
        import torch
        if state['frames'] != 634:
            raise ValueError('incremental frame coverage mismatch')
        timeline = torch.cat(state['latents'], dim=2).to(device=ops.device)
        with ops.inference(), ops.autocast():
            reference = ops.pixels(stage.vae, stage.vae._decode(timeline))
        incremental = torch.cat(state['pixels'], dim=2)
        if reference.shape != incremental.shape or reference.shape[2] != 634:
            raise ValueError('reference frame coverage mismatch')
        diff = (reference.float() - incremental.float()).abs()
        per = diff.permute(2, 0, 1, 3, 4).reshape(634, -1)
        receipt.update(frame_count=634, reference_frames=634, exact=bool(torch.equal(reference, incremental)),
                       max=float(diff.max()), mean=float(diff.mean()),
                       per_frame_max=per.max(dim=1).values.tolist(), per_frame_mean=per.mean(dim=1).tolist())
    _save(c, ops.rank, receipt)
    state['latents'].clear()
    state['pixels'].clear()
    state['final'] = True
