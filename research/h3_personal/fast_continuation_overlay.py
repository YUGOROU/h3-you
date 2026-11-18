"""Opt-in overlays on an isolated, already-audited SMD4/locked-audio copy.

No source archive, installed package, model, scheduler or live service is changed
on import. Install all anchors transactionally, then use a separate per-window
manifest; independent mode retains the existing warmup and generation branches.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = 'fastvideo/pipelines/basic/minimax_h3/stages/'
KEY = '_h3_continuation'


def install(source):
    source = Path(source)
    latent = source / (ROOT+'minimax_h3_latent_preparation.py')
    denoise = source / (ROOT+'minimax_h3_denoising.py')
    decode = source / (ROOT+'minimax_h3_decoding.py')
    originals = {p: p.read_text() for p in (latent, denoise, decode)}
    updated = dict(originals)

    def change(path, old, new):
        text = updated[path]
        if text.count(old) != 1:
            raise ValueError('continuation overlay source anchor differs')
        updated[path] = text.replace(old, new)

    if any('fast_continuation_' in t for t in originals.values()):
        raise ValueError('continuation overlay is single-use')
    # Packing remains pristine, separate from the controlled stage overlays.
    from .fast_continuation import PACKING_SHA, REFERENCE_SHA
    base = source/'fastvideo/pipelines/basic/minimax_h3'
    for filename, expected in (('packing.py', PACKING_SHA), ('reference.py', REFERENCE_SHA)):
        if hashlib.sha256((base/filename).read_bytes()).hexdigest() != expected:
            raise ValueError('continuation overlay requires pinned FastVideo source')
    anchor = '        if not video_rows:\n'
    change(latent, anchor,
        "        from research.h3_personal.fast_continuation_stages import active, append_conditions\n"
        "        if active():\n"
        "            video_rows, audio_rows = append_conditions(self, batch, fastvideo_args, references, video_rows, audio_rows)\n"+anchor)
    anchor = '        text_token_tags = batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)\n'
    change(latent, anchor,
        "        if '_h3_continuation' in batch.extra:\n"
        "            from research.h3_personal.fast_continuation_stages import make_layout\n"
        "            return make_layout(self, batch, fastvideo_args)\n"+anchor)
    anchor = ("        if __import__('os').environ.get('H3_SMD_LOCKED_AUDIO') == '1':\n"
              "            from research.h3_personal.external_audio import lock_target_audio_rows\n"
              "            batch = lock_target_audio_rows(self, batch, fastvideo_args)\n")
    change(latent, anchor,
        "        if '_h3_continuation' in batch.extra:\n"
        "            from research.h3_personal.fast_continuation_stages import bind_audio\n"
        "            batch = bind_audio(self, batch, fastvideo_args)\n"+
        anchor.replace('        if ', '        elif ', 1))
    anchor = '                    # Stop if interrupted\n'
    change(denoise, anchor,
        "                    if '_h3_continuation' in batch.extra:\n"
        "                        from research.h3_personal.fast_continuation_stages import before_forward\n"
        "                        before_forward(self, batch, index)\n"+anchor)
    anchor = ("                    if __import__('os').environ.get('H3_SMD_LOCKED_AUDIO') == '1':\n"
              "                        from research.h3_personal.external_audio import hold_locked_audio_rows\n"
              "                        hold_locked_audio_rows(self, batch, audio_velocity)\n")
    change(denoise, anchor,
        "                    if '_h3_continuation' in batch.extra:\n"
        "                        from research.h3_personal.fast_continuation_stages import hold_audio\n"
        "                        hold_audio(self, batch, audio_velocity)\n"+
        anchor.replace('                    if ', '                    elif ', 1))
    # Exact pinned native model invocation, excluding scheduler/hold work.
    anchor = '''                        video_velocity, audio_velocity = self.transformer(
                            hidden_states=batch.latents[None],
                            audio_hidden_states=batch.audio_latents[None],
                            encoder_hidden_states=prompt_embeds,
                            timestep=unique_timesteps,
                            timestep_indices=timestep_indices,
                            token_tags=token_tags,
                            position_ids=position_ids,
                            video_indices=video_indices,
                            audio_indices=audio_indices,
                            text_indices=text_indices,
                        )
'''
    change(denoise, anchor,
        '                        from research.h3_personal.fast_continuation_stages import forward_profile\n'
        '                        with forward_profile(batch, index):\n' +
        ''.join('    ' + line for line in anchor.splitlines(keepends=True)))
    # Both the target clean timestep and trained SMD4 schedule must already exist.
    if ('audio_value = locked_audio_timestep(audio_value)' not in updated[denoise] or
            'configure_schedulers(self.scheduler, self.audio_scheduler, batch.num_inference_steps, device)' not in updated[denoise]):
        raise ValueError('existing SMD4/locked-audio overlays required')
    anchor = '        return batch\n'
    change(denoise, anchor,
        "        if '_h3_continuation' in batch.extra:\n"
        "            from research.h3_personal.fast_continuation_stages import commit\n"
        "            batch = commit(self, batch, fastvideo_args)\n"+anchor)
    anchor = '        placeholder = torch.empty((0, 3, 0, 0, 0), device="cpu", dtype=torch.float32)\n'
    change(decode, anchor,
        "        if '_h3_continuation' in batch.extra:\n"
        "            from research.h3_personal.fast_continuation_decode import decode_suffix\n"
        "            return decode_suffix(self, batch, fastvideo_args)\n"+anchor)
    # Compile and validate EVERYTHING before touching the isolated copy.
    for p, text in updated.items():
        compile(text, str(p), 'exec')
    reports = {}
    for p, text in updated.items():
        temporary = p.with_suffix('.continuation.tmp')
        temporary.write_text(text)
        os.replace(temporary, p)
        reports[p.name] = dict(original_sha256=hashlib.sha256(originals[p].encode()).hexdigest(),
                               patched_sha256=hashlib.sha256(text.encode()).hexdigest())
    return dict(mode='latent-tail-v1-opt-in', stages=reports, gpu_verified=False)


def finish_batch(batch):
    """Persist scalar decode receipt BEFORE native cleanup/IPC. No tensor export."""
    context = batch.extra.get(KEY)
    if context is None:
        return batch
    import torch
    from .fast_continuation_decode import RECEIPT_KEY
    receipt = batch.extra.get(RECEIPT_KEY)
    if receipt is None:
        raise ValueError('continuation decode receipt missing')
    m = context['manifest']
    rank = torch.distributed.get_rank()
    row = dict(receipt, session_id=m['session_id'], window_index=m['index'],
               worker_rank=rank, total_frames=m['total_frames'], gpu_verified=False)
    root = Path(os.environ['H3_REF2VA_COVERAGE_DIR'])
    path = root/f'h3-continuation-decode-rank-{rank}-window-{m["index"]}.json'
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(row, stream)
    os.replace(temporary, path)
    # The source/guide owner stays on its worker stages. Never pickle it or its
    # private tensors through ForwardBatch.extra with the generated RGB output.
    batch.extra.pop(KEY)
    batch.extra.pop(RECEIPT_KEY)
    return batch
