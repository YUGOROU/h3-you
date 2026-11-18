"""Fixed generated-audio continuation diagnostic; offline unless explicitly requested.

--prepare stages network-checked metadata only. --submit delegates one finite Job
and locked accounting to the shared launcher. No live input, TTS or device access.
The supplied ledger must contain a reconciled authorized_cap_usd for this scope.
"""
import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import FunctionType
import uuid

from research import launch_h3_fast_ref2va as base
from research import prepare_h3_fast_ref2va_gpu as staging
from research.launch_h3_realtime_first import main as launch
from research.h3_personal import continuation_diagnostic as diagnostic
from research.h3_personal import continuation_candidate, continuation_efficiency
from research.h3_personal.continuation_plan import plan
from research.prepare_h3_realtime_first import safe_path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = 'research/bootstrap_h3_fast_ref2va.py'
WORKER = 'research/hf_h3_continuation_diagnostic.py'
EXTRA = (WORKER, 'research/hf_h3_conditioning_exploration.py',
         'research/h3_personal/accepted_profiles.py',
         'research/h3_personal/av_delivery.py',
         'research/h3_personal/incremental_av.py',
         'research/h3_personal/headless_av_sink.py',
         'research/h3_personal/conversation_state.py',
         'research/h3_personal/buffered_stream.py',
         'research/h3_personal/component_cache.py',
         'research/h3_personal/conditioning_experiment.py',
         'research/hf_h3_efficiency_comparison.py',
         'research/hf_h3_utilization_comparison.py',
         'research/h3_personal/device_activity.py',
         'research/h3_personal/local_denoise_compile.py',
         'research/h3_personal/denoise_execution.py',
         'research/h3_personal/utilization_policy.py',
         'research/h3_personal/parallel_decode_parity.py',
         'research/h3_personal/live_voice_runtime.py',
         'research/h3_personal/continuation_diagnostic.py',
         'research/h3_personal/continuation_plan.py',
         'research/h3_personal/continuation_candidate.py',
         'research/h3_personal/continuation_efficiency.py',
         'research/h3_personal/fast_continuation.py',
         'research/h3_personal/fast_continuation_stages.py',
         'research/h3_personal/fast_continuation_decode.py',
         'research/h3_personal/fast_continuation_overlay.py',
         'research/h3_personal/fast_continuation_render.py',
         'research/h3_personal/incremental_decode.py',
         'research/h3_personal/decode_parity.py',
         'research/h3_personal/decode_parity_capture.py',
         'research/h3_personal/adapter_fingerprint.py')
APPROVAL = ('Explicit new USER plan: explore geometry/conditioning as a separate immutable '
            'experiment, no production promotion; allow private generated review uploads, '
            'never inputs, latents or credentials. Same full-rate cap, finite USER budget '
            'and all shared permissions/location/license/idle/budget gates. '
            'USER go, GPU post-operation reporting: within existing HF ~2.5h h200x2 '
            'allocation, one finite Job for diagnostic inference/artifact validation; '
            'optional immutable independent-supplied-speech/continuation comparison or '
            'same-committed-latent serial decoder parity, or immutable one-factor '
            'window-audio-context quality candidate with separately toggled receipt I/O, '
            'or precision-preserving parallel/device/I/O efficiency acceptance and '
            'profile-OFF comparison, or bounded one-window metadata-only dual-rank trace '
            'and local post-attention compile adoption comparison, in one finite Job '
            'on the same warm runtime under current plan/patch; '
            'retains tensors in memory only and uses one warm runtime with explicitly '
            'reused generated diagnostic audio. '
            'Submit after fresh permission/license/location/pin/concurrency and supplied '
            'ledger checks; report Job status and costs afterward. No Discord, devices, '
            'live input, TTS or Endpoint. No timeout renewal or provider/quality fallback.')


def contract(timeout):
    source = diagnostic.source_contract()
    planned = dict(schema='h3-smd-continuation-plan-v1', mode='latent-tail-v1',
                   sha256=source['sha256'], source_samples=source['source_samples'],
                   source_sample_rate=source['source_sample_rate'], source_channels=1,
                   total_frames=328, window_frames=124, overlap_frames=22,
                   expected_windows=3, sampler='SMD4', fps=24, seed=314159,
                   windows=[w.record() for w in plan(328, 124, 22)])
    return diagnostic.validate_metadata(source, planned, timeout_seconds=timeout)


def decode_parity_contract():
    return dict(schema='h3-same-latent-decode-parity-v1', serial=True,
                atol=0.0, retain_latents_in_memory_only=True,
                total_frames=328, arrivals=[37, 30, 30])


def locked_speech():
    source = diagnostic.source_contract()
    return dict(repo=source['repo'], revision=source['revision'], path=source['path'],
                sha256=source['sha256'], sample_rate=source['source_sample_rate'],
                samples=source['source_samples'])


def budget_for(ledger):
    # Re-read the caller's authority on every invocation, including under submit lock.
    def bound(*, ledger_bound, timeout_seconds, hourly_rate=Decimal('10.00002')):
        contract(timeout_seconds)
        cap = Decimal(str(json.loads(Path(ledger).read_text())['authorized_cap_usd']))
        total, rate = Decimal(str(ledger_bound)), Decimal(str(hourly_rate))
        if not all(v.is_finite() for v in (cap, total, rate)) or total < 0 or cap <= 0 or not 0 < rate <= Decimal('10.00002'):
            raise ValueError('fresh ledger authority and bounded h200x2 rate required')
        cost = rate * Decimal(timeout_seconds) / Decimal(3600)
        if total + cost > cap:
            raise ValueError('supplied ledger authority exceeded')
        return dict(job_bound=cost, cumulative_bound=total+cost,
                    remaining=cap-total-cost, authorized_cap=cap)
    return bound


def inventory(stage):
    rows = []
    for path in sorted(stage.rglob('*')):
        if path.is_symlink():
            raise ValueError('deployment symlink forbidden')
        if path.is_file() and path != stage/'upload-allowlist.json':
            rows.append(dict(path=path.relative_to(stage).as_posix(),
                             bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(schema=base.SCOPE, private_only=True, files=rows)


def write_inventory(stage):
    (stage/'upload-allowlist.json').write_text(json.dumps(inventory(stage), indent=2))


@contextmanager
def base_view(stage):
    """Verify original bytes, then adapt ONLY the legacy inspector's timeout.

    A private ephemeral copy avoids changing the deployment or monkeypatching shared
    modules. All base model/license/source pins and remote permission checks survive.
    """
    stage = Path(stage)
    recorded = json.loads((stage/'upload-allowlist.json').read_text())
    for row in recorded['files']:
        safe_path(row['path'])
    if recorded['files'] != inventory(stage)['files']:
        raise ValueError('deployment bytes/inventory differ')
    spec = json.loads((stage/'spec.json').read_text())
    seconds = spec['config']['timeout_seconds']
    contract(seconds)
    if spec['execution']['timeout_seconds'] != seconds:
        raise ValueError('execution timeout differs')
    with tempfile.TemporaryDirectory(prefix='h3-diagnostic-inspect-') as directory:
        view = Path(directory)/'deployment'
        shutil.copytree(stage, view)
        projected = json.loads(json.dumps(spec))
        projected['config']['timeout_seconds'] = min(seconds, 1500)
        projected['execution']['timeout_seconds'] = min(seconds, 1500)
        (view/'spec.json').write_text(json.dumps(projected))
        write_inventory(view)
        yield view, spec, recorded


def inspect_base(stage):
    with base_view(stage) as (view, spec, recorded):
        base.inspect(view)
    return spec, recorded


def inspect(stage):
    spec, recorded = inspect_base(stage)
    expected = contract(spec['config']['timeout_seconds'])
    from research.hf_h3_continuation_diagnostic import validate_conditioning_mode
    validate_conditioning_mode(spec)
    if 'utilization_comparison' in spec:
        from research.hf_h3_utilization_comparison import comparison_contract, EXCLUSIVE
        from research.hf_h3_continuation_diagnostic import strict_equal
        if (any(key in spec for key in EXCLUSIVE) or
                not strict_equal(spec['utilization_comparison'], comparison_contract())):
            raise ValueError('invalid or incompatible utilization comparison')
    if 'efficiency_comparison' in spec:
        if any(key in spec for key in ('continuation_candidate', 'continuation_comparison',
                                       'decode_parity_diagnostic')):
            raise ValueError('efficiency comparison and other modes are mutually exclusive')
        if json.dumps(spec['efficiency_comparison'], sort_keys=True) != json.dumps(
                continuation_efficiency.comparison_contract(), sort_keys=True):
            raise ValueError('efficiency comparison contract differs')
    if 'continuation_candidate' in spec:
        options = continuation_candidate.normalize(spec['continuation_candidate'])
        if options is None or options == dict(
                schema='h3-continuation-candidate-v1', audio_context='full-source-v1',
                progress_receipts='per-step-v1'):
            raise ValueError('continuation candidate requires actual change from legacy')
        if any(key in spec for key in ('continuation_comparison', 'decode_parity_diagnostic')):
            raise ValueError('candidate and exclusive diagnostics are mutually exclusive')
    if 'decode_parity_diagnostic' in spec:
        if 'continuation_comparison' in spec:
            raise ValueError('decode parity and comparison are mutually exclusive')
        if json.dumps(spec['decode_parity_diagnostic'], sort_keys=True) != json.dumps(decode_parity_contract(), sort_keys=True):
            raise ValueError('decode parity diagnostic contract differs')
    if 'continuation_comparison' in spec:
        from research.hf_h3_continuation_diagnostic import CONTINUATION_COMPARISON
        if json.dumps(spec['continuation_comparison'], sort_keys=True) != json.dumps(CONTINUATION_COMPARISON, sort_keys=True):
            raise ValueError('paired comparison contract differs')
    if json.dumps(spec.get('continuation_diagnostic'), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('fixed three-window diagnostic contract differs')
    if spec.get('locked_speech') != locked_speech():
        raise ValueError('fixed generated diagnostic speech required')
    if any(key in spec for key in ('live_voice_session', 'asr_policy', 'modal_live_approval')):
        raise ValueError('no live/ASR/Modal authority in diagnostic')
    if (spec.get('layout'), spec.get('runtime_policy'), spec.get('attention_mode')) != ('portrait', 'resident-visual-cached', 'fa4-packed') or any(spec.get(k) for k in ('regional_compile', 'warm_stage_diagnostic', 'incremental_transfer', 'face_crop_box')):
        raise ValueError('fixed BF16 SMD4 resident portrait policy required')
    names = {r['path'] for r in recorded['files']}
    if not set(EXTRA) <= names or 'tts-manifest.json' in names or 'research/hf_h3_live_voice.py' in names:
        raise ValueError('dedicated diagnostic runtime required; no live/TTS worker')
    bootstrap = (Path(stage)/BOOTSTRAP).read_text()
    if 'hf_h3_continuation_diagnostic' not in bootstrap or 'continuation_diagnostic' not in bootstrap:
        raise ValueError('bootstrap continuation dispatch is not integrated')
    return spec, recorded


def pre_submit(api, spec, stage):
    checked, _ = inspect(stage)
    if checked != spec:
        raise ValueError('submission spec differs')
    with base_view(stage) as (view, _, _):
        projected, _ = base.inspect(view)
        base.pre_submit(api, projected, view)


def prepare(directory, template, ledger, api, timeout=1800, *, comparison=False,
            decode_parity=False, window_audio_context=False, defer_progress_receipts=False,
            efficiency_comparison=False, utilization_comparison=False, conditioning_exploration=False):
    if any(type(value) is not bool for value in (comparison, decode_parity, efficiency_comparison, utilization_comparison, conditioning_exploration)):
        raise ValueError('explicit boolean diagnostic opt-ins required')
    options = continuation_candidate.make_options(
        window_audio_context=window_audio_context,
        defer_progress_receipts=defer_progress_receipts)
    if (sum((comparison, decode_parity, efficiency_comparison, utilization_comparison, conditioning_exploration)) > 1 or
            (options is not None and (comparison or decode_parity or efficiency_comparison or utilization_comparison or conditioning_exploration))):
        raise ValueError('candidate, decode parity and comparison are mutually exclusive')
    contract(timeout)
    stage = Path(directory).resolve()/'deployment'
    if stage.exists():
        raise ValueError('fresh deployment required')
    for name in EXTRA:
        if not (ROOT/name).is_file():
            raise ValueError('runtime integration missing: '+name)
    # The base preparer has no dependency-injection parameters. Bind a private
    # function copy to our budget/timeout adapters; never mutate shared globals.
    dependencies = dict(staging.prepare.__globals__, budget_bound=budget_for(ledger),
                        inspect=inspect_base)
    preparer = FunctionType(staging.prepare.__code__, dependencies,
                            staging.prepare.__name__, staging.prepare.__defaults__)
    result = preparer(stage, template, ledger, api,
                      publication_repo='YUGOROU/h3-fast-ref2va-smd4-gpu-continuation-'+uuid.uuid4().hex[:12],
                      timeout_seconds=timeout, attention_mode='fa4-packed',
                      layout='portrait', runtime_policy='resident-visual-cached',
                      locked_speech=locked_speech())
    for name in EXTRA:
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    spec = json.loads((stage/'spec.json').read_text())
    spec['continuation_diagnostic'] = contract(timeout)
    if conditioning_exploration:
        from research.hf_h3_conditioning_exploration import CONTRACT
        spec['conditioning_exploration'] = json.loads(json.dumps(CONTRACT))
    if utilization_comparison:
        from research.hf_h3_utilization_comparison import comparison_contract
        spec['utilization_comparison'] = comparison_contract()
    if efficiency_comparison:
        spec['efficiency_comparison'] = continuation_efficiency.comparison_contract()
    if options is not None:
        spec['continuation_candidate'] = options
    if decode_parity:
        spec['decode_parity_diagnostic'] = decode_parity_contract()
    if comparison:
        from research.hf_h3_continuation_diagnostic import CONTINUATION_COMPARISON
        spec['continuation_comparison'] = CONTINUATION_COMPARISON
    (stage/'spec.json').write_text(json.dumps(spec, indent=2))
    write_inventory(stage)
    spec, _ = inspect(stage)
    pre_submit(api, spec, stage)  # Permissions/pins only; never launch here.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--submit', action='store_true')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--timeout', type=int, default=1800)
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument('--comparison', action='store_true', help='prepare immutable paired warm comparison')
    diagnostics.add_argument('--decode-parity', action='store_true', help='prepare immutable same-latent decode parity diagnostic')
    diagnostics.add_argument('--efficiency-comparison', action='store_true',
                             help='prepare precision-preserving warm efficiency comparison')
    diagnostics.add_argument('--utilization-comparison', action='store_true',
                             help='prepare bounded trace and local compile comparison')
    diagnostics.add_argument('--conditioning-exploration', action='store_true',
                             help='prepare isolated immutable geometry/conditioning experiment')
    parser.add_argument('--window-audio-context', action='store_true')
    parser.add_argument('--defer-progress-receipts', action='store_true')
    args = parser.parse_args(argv)
    candidate = args.window_audio_context or args.defer_progress_receipts
    if candidate and (args.comparison or args.decode_parity or args.efficiency_comparison or args.utilization_comparison or args.conditioning_exploration):
        parser.error('candidate switches cannot be combined with comparison or decode parity')
    if (candidate or args.comparison or args.decode_parity or args.efficiency_comparison or args.utilization_comparison or args.conditioning_exploration) and not args.prepare:
        parser.error('diagnostic switches are prepare-only; submission uses the immutable staged contract')
    contract(args.timeout)
    if args.prepare:
        if args.template is None:
            parser.error('--prepare requires --template')
        from huggingface_hub import HfApi
        from research.provision_modal_identity_secret import cached_token
        print(json.dumps(prepare(args.directory, args.template, args.ledger,
                                 HfApi(token=cached_token()), args.timeout,
                                 comparison=args.comparison, decode_parity=args.decode_parity,
                                 efficiency_comparison=args.efficiency_comparison,
                                 utilization_comparison=args.utilization_comparison,
                                 conditioning_exploration=args.conditioning_exploration,
                                 window_audio_context=args.window_audio_context,
                                 defer_progress_receipts=args.defer_progress_receipts)))
        return 0
    if args.comparison or args.decode_parity:
        parser.error('--comparison and --decode-parity are prepare-only; submission uses the immutable staged contract')
    stage = args.directory/'deployment'
    spec, _ = inspect(stage)
    if spec['config']['timeout_seconds'] != args.timeout:
        raise ValueError('caller timeout differs from immutable deployment')
    arguments = ['--stage', str(stage), '--ledger', str(args.ledger)]
    if args.submit:
        arguments.append('--submit')
    return launch(arguments, inspect_fn=inspect, pre_submit=pre_submit,
                  budget_fn=budget_for(args.ledger), bootstrap_entry=BOOTSTRAP,
                  approval_request=APPROVAL)


if __name__ == '__main__':
    raise SystemExit(main())
