"""Explicit resident eager decoder weight-norm experiment; never a default switch.

Pinned source: OpenBMB/VoxCPM cce58a5b59303c9bd63d12f23afe6a49a4a80c59,
modules/audiovae/audio_vae_v2.py: streaming_decode owns a decoder context;
decode_chunk accepts [batch, latent_dim, time]. No installs or synchronization.
Failure after folding leaves the resident model modified: discard it, do not
silently use it in production. The supplied measure must implement the bounded
h3_personal.tts_streaming_measure.measure contract (including full exhaustion).
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time


def _targets(decoder, torch):
    cls = importlib.import_module('torch.nn.utils.weight_norm').WeightNorm
    selected = []
    for name, module in decoder.named_modules():
        hooks = [h for h in module._forward_pre_hooks.values() if isinstance(h, cls)]
        if any(h.name != 'weight' for h in hooks) or len(hooks) > 1:
            raise RuntimeError('unsupported legacy weight_norm hooks')
        if hooks:
            selected.append((name, module))
    return selected


def _decode(vae, patches, torch):
    with torch.inference_mode(), vae.streaming_decode() as decoder:
        return [decoder.decode_chunk(z.clone()).detach().clone() for z in patches]


def run(candidate, text, out, torch, measure):
    """Six bounded drains: warm + two baseline, exact gate, warm + two folded.

    Seeds reset before every drain. Warm media live only in temporary directories.
    Host times are not GPU timings; native EOS remains deliberately unknown.
    Exact latent equivalence is necessary, not a voice-quality acceptance claim.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError('require nonempty text of at most 4096 characters')
    if candidate.observe_stop is not False or candidate._active or sys.gettrace() is not None:
        raise RuntimeError('idle candidate and trace OFF required')
    model = candidate._model.tts_model
    modules = list(model.modules())
    if any(m.training or hasattr(m, '_orig_mod') or hasattr(m, '_compiled_call_impl')
           and m._compiled_call_impl is not None for m in modules):
        raise RuntimeError('eval-only EAGER model required')
    vae = model.audio_vae
    patch = model.patch_size
    if type(patch) is not int or not 1 <= patch <= 128 or not 1 <= vae.latent_dim <= 4096:
        raise RuntimeError('unsupported latent bounds')
    selected = _targets(vae.decoder, torch)
    if not selected:
        raise RuntimeError('no legacy decoder weight_norm to diagnose')
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(experiment='decoder-legacy-weight-norm-fold-only', status='running',
                  production_adoption=False, native_eos='unknown', cases=[],
                  selected_modules=[name for name, _ in selected],
                  selected_count=len(selected), exact_pcm_equal=False)
    import numpy as np  # only the caller's isolated model environment

    def call(label, destination):
        np.random.seed(42)
        torch.manual_seed(42)
        start = time.monotonic()
        with torch.inference_mode():
            result = measure(candidate, text, destination, 'float', timing_attribution=True)
        return dict(case=label, settings=dict(seed=42, eager=True, trace=False,
                    decoder_folded=label.startswith('folded'), kind='float'),
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    wall_seconds=time.monotonic()-start, measurement=result,
                    native_eos='unknown')

    try:
        with tempfile.TemporaryDirectory(prefix='voxcpm-warm-') as temp:
            call('baseline-warm', Path(temp)/'warm')
        for i in range(2):
            report['cases'].append(call('baseline-'+str(i), directory/('baseline-'+str(i))))
        parameter = next(vae.parameters())
        generator = torch.Generator(device='cpu').manual_seed(42)
        z = torch.randn(1, vae.latent_dim, 2*patch, generator=generator,
                        device='cpu', dtype=torch.float32).to(device=parameter.device, dtype=parameter.dtype)
        patches = [z[..., :patch].clone(), z[..., patch:].clone()]
        before = _decode(vae, patches, torch)
        # Removed g/v parameters are replaced with derived weight; all surviving
        # learned parameter objects must retain dtype. No tensor casts or writes.
        dtypes = {id(p): p.dtype for p in model.parameters()}
        selected_dtypes = {name: module.weight.dtype for name, module in selected}
        for _, module in selected:
            torch.nn.utils.remove_weight_norm(module, name='weight')
        report['remaining_hooks'] = len(_targets(vae.decoder, torch))
        report['parameter_dtypes_unchanged'] = (
            all(p.dtype == dtypes[id(p)] for p in model.parameters() if id(p) in dtypes)
            and all(module.weight.dtype == selected_dtypes[name] for name, module in selected))
        if report['remaining_hooks'] or not report['parameter_dtypes_unchanged']:
            raise RuntimeError('fold hook/dtype validation failed')
        after = _decode(vae, patches, torch)
        report['exact_pcm_equal'] = all(torch.equal(a, b) for a, b in zip(before, after))
        report['latent_frames'] = 2*patch
        report['pcm_shapes'] = [list(a.shape) for a in before]
        if not report['exact_pcm_equal']:
            raise RuntimeError('exact streaming PCM equivalence failed; discard resident model')
        with tempfile.TemporaryDirectory(prefix='voxcpm-warm-') as temp:
            call('folded-warm', Path(temp)/'warm')
        for i in range(2):
            report['cases'].append(call('folded-'+str(i), directory/('folded-'+str(i))))
        report['status'] = 'completed-experiment-only'
        return report
    except BaseException as exc:
        report.update(status='blocked', blocker=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
