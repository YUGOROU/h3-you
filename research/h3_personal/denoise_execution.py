"""Main50 instance controller; no device/network imports.

Worker contract: status.min(bool, label) is a synchronous BOTH-rank MIN on a
bounded process group. Worker owns an external hard process deadline (Python
cannot interrupt a compiler or a failed native/NCCL collective). Call run() on
BOTH ranks, in identical order. Never reuse a faulted owner. No eager fallback.
"""
import time
import math
import types
import hashlib
import logging

PREPARE_ERROR_CODES = ('ffn-configuration-unsupported',
                       'native-provenance-unsupported',
                       'parameter-dtype-unsupported')
from contextlib import contextmanager
from . import local_denoise_compile as local


class _AgreementRejected(RuntimeError):
    """MIN completed successfully and reported failure."""


class CompileBudget:
    """Counts actual backend invocations, including Dynamo recompiles."""
    def __init__(self, backend, max_count, max_seconds, clock=time.monotonic):
        if (not callable(backend) or type(max_count) is not int or max_count < 1
                or type(max_seconds) not in (int, float)
                or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('finite compile budget required')
        self.backend, self.max_count, self.max_seconds = backend, max_count, max_seconds
        self.clock = clock
        self.count = 0
        self.seconds = 0.0

    def __call__(self, *args, **kwargs):
        if self.count >= self.max_count or self.seconds >= self.max_seconds:
            raise CandidateRejected('compile budget exhausted')
        self.count += 1
        start = self.clock()
        try:
            result = self.backend(*args, **kwargs)
        finally:
            self.seconds += self.clock() - start
        if self.seconds > self.max_seconds:
            raise CandidateRejected('compile time exceeded; external deadline required')
        return result


def _native_fp32_roots(source, model):
    """Read only the actual pinned class's literal loader declaration; no eval."""
    import ast
    classes = [n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == type(model).__name__]
    declarations = [n.value for cls in classes for n in cls.body
                    if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == '_keep_in_fp32_modules'
                        for t in n.targets)]
    actual = getattr(model, '_keep_in_fp32_modules', ())
    if not declarations:
        if actual:
            raise ValueError('missing native FP32 declaration')
        return frozenset()
    if len(classes) != 1 or len(declarations) != 1:
        raise ValueError('ambiguous native FP32 declaration')
    value = declarations[0]
    if isinstance(value, ast.Call):
        if (not isinstance(value.func, ast.Name)
                or value.func.id not in ('frozenset', 'set', 'list', 'tuple')
                or len(value.args) != 1 or value.keywords):
            raise ValueError('nonliteral native FP32 declaration')
        value = value.args[0]
    if (not isinstance(value, (ast.Set, ast.List, ast.Tuple))
            or any(not isinstance(v, ast.Constant) or type(v.value) is not str
                   for v in value.elts)):
        raise ValueError('literal native FP32 roots required')
    roots = frozenset(v.value for v in value.elts)
    if (type(actual) not in (frozenset, set, list, tuple)
            or any(type(v) is not str for v in actual)
            or frozenset(actual) != roots
            or any(not v.isidentifier() for v in roots)):
        raise ValueError('actual native FP32 roots differ from pinned class')
    return roots


class DenoiseController:
    """Explicit main-stack adapter: pass model's actual main50 (not refiner2).

    gates must attest actual static BF16 weights/merged adapter and source pin.
    Identity receipt contains names/ids/metadata only, never weight values.
    Shape metadata is supplied from actual worker packing, never inferred R/T.
    """
    _attrs = ('forward', '_h3_local_compile_receipt', '_h3_region_b_dispatch',
              '_h3_local_post_attention', '_h3_local_stage', '_h3_local_forward_index',
              '_h3_local_probe')

    def __init__(self, model, main_blocks, *, status, compiler, backend,
                 max_compile_count, max_compile_seconds, clock=time.monotonic):
        self.model, self.blocks = model, tuple(main_blocks)
        self.status, self.compiler, self.clock = status, compiler, clock
        self.budget = CompileBudget(backend, max_compile_count, max_compile_seconds, clock)
        self.state = 'new'
        self.context = None
        self.receipt = {'agreements': 0, 'agreement_seconds': 0.,
                        'baseline_extra_agreements': 0,
                        'measurement_extra_agreements': 0,
                        'native_pre_attention_failure': 'external bounded process deadline',
                        'compiler_interruptible': False}

    def _agree(self, ok, label):
        start = self.clock()
        try:
            accepted = self.status.min(bool(ok), label)
        except BaseException:
            self.state = 'faulted'
            raise
        finally:
            self.receipt['agreements'] += 1
            self.receipt['agreement_seconds'] += self.clock() - start
        if not accepted or not ok:
            self.state = 'faulted'
            raise _AgreementRejected('denoise rank agreement failed: ' + label)

    def _finish_boundary(self, error, label):
        return self._post_agreement(error, label)

    def _post_agreement(self, error, label):
        # Healthy path retains exactly one MIN. Only a failed MIN enters MAX.
        try:
            self._agree(error is None, label)
        except _AgreementRejected as agreement_error:
            vote = getattr(self.status, 'rejection_vote', None)
            if not callable(vote):
                raise RuntimeError('status requires rejection_vote MAX API') from agreement_error
            code = 0 if error is None else (1 if _controlled(error) else 2)
            outcome = vote(code, label)
            if outcome == 1:
                self.receipt['candidate_rejected'] = True
                self.receipt['rejection_kind'] = 'controlled-local-rejection'
                raise CandidateRejected('both-rank controlled rejection: ' + label) from error
            self.receipt['candidate_rejected'] = False
            raise RuntimeError('unexpected denoise failure: ' + label) from error

    def _identity(self):
        params = tuple((n, id(p), str(p.dtype), tuple(p.shape))
                       for n, p in self.model.named_parameters())
        # keep_vars avoids even detached tensor wrappers; no tensor/model copy.
        state = tuple((n, id(p)) for n, p in self.model.state_dict(keep_vars=True).items())
        return params, state

    def install(self, *, source, namespace, admission, stage_shapes, gates):
        if self.state != 'new':
            raise RuntimeError('owner refuses install')
        snapshots = []
        error = None
        self.receipt.update(source_pinned_verified=False, main50_verified=False,
                            prepare_error_code=None)
        try:
            # Retain actual identity even when a later prepare guard fails.
            before = self.identity = self._identity()
            self.receipt['identity_unchanged'] = True
            self.receipt['native_type_hashes'] = sorted(set(
                hashlib.sha256((type(obj).__module__ + '.' + type(obj).__qualname__)
                               .encode()).hexdigest()
                for obj in (self.model, *self.blocks)))
            required = {'source_pinned': True, 'real_bf16_weights': True,
                        'static_adapter_receipt_verified': True, 'main50_verified': True}
            if (type(gates) is not dict or gates.keys() != required.keys()
                    or any(gates[k] is not True for k in required)):
                raise ValueError('caller weight/adapter/main-stack gates required')
            # SHA validation precedes ALL source-derived precision admission.
            code = local.extract(source)
            self.receipt['source_pinned_verified'] = True
            if len(self.blocks) != 50 or len({id(b) for b in self.blocks}) != 50:
                raise ValueError('exactly 50 distinct main blocks required')
            self.receipt['main50_verified'] = True
            declared = _native_fp32_roots(source, self.model)
            known_dtypes = {'torch.' + d for d in (
                'bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16',
                'int32', 'int64', 'uint8', 'bool', 'complex64', 'complex128',
                'float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz')}
            histogram = {}
            for _, _, dtype, _ in before[0]:
                if dtype not in known_dtypes:
                    raise ValueError('unknown parameter dtype metadata')
                histogram[dtype] = histogram.get(dtype, 0) + 1
            self.receipt['parameter_dtype_histogram'] = histogram
            dtype_error = None
            for name, _, dtype, shape in before[0]:
                if dtype == 'torch.bfloat16':
                    continue
                root = name.split('.', 1)[0]
                # Native loader exemptions are protected baseline, including
                # input/output matrices. Main-stack weights remain BF16.
                if not (dtype == 'torch.float32' and root in declared
                        and root != 'transformer_blocks'):
                    dtype_error = local.AdmissionUnsupported('parameter-dtype-unsupported',
                        'actual native BF16/FP32 loader parameters required')
            scope = dict(namespace)
            exec(code, scope)
            prepared = (code, scope['_h3_region_b'], scope['forward'])
            shared = None

            def factory(helper, **settings):
                nonlocal shared
                if shared is None:
                    shared = self.compiler(helper, backend=self.budget, **settings)
                return shared

            commits = []
            for i, block in enumerate(self.blocks):
                snapshots.append({k: block.__dict__[k] for k in self._attrs if k in block.__dict__})
                try:
                    commits.append(local.install(block, source=source, namespace=namespace,
                        enabled=True, main_block_index=i, admission=admission,
                        stage_shapes=stage_shapes, compile_factory=factory,
                        prepare_only=True, prepared_source=prepared))
                except local.AdmissionUnsupported as exc:
                    if exc.code not in PREPARE_ERROR_CODES:
                        raise
                    dtype_error = exc
            if dtype_error is not None:
                raise dtype_error
        except Exception as exc:
            error = exc
            logging.getLogger(__name__).error('local denoise prepare failed', exc_info=True)
        if hasattr(self, 'identity'):
            self.receipt['identity_unchanged'] = self._identity() == self.identity
        controlled = (type(error) is local.AdmissionUnsupported
            and error.code in PREPARE_ERROR_CODES
            and self.receipt['source_pinned_verified'] is True
            and self.receipt['main50_verified'] is True
            and self.receipt.get('identity_unchanged') is True)
        # One common MIN on BOTH ranks; only failure enters the typed MAX.
        try:
            self._agree(error is None, 'prepare-main50')
        except _AgreementRejected:
            code = (PREPARE_ERROR_CODES.index(error.code) + 1 if controlled
                    else 0 if error is None else 100)
            outcome = self.status.rejection_vote(code, 'prepare-main50')
            if (outcome in range(1, len(PREPARE_ERROR_CODES) + 1)
                    and self.receipt['source_pinned_verified'] is True
                    and self.receipt['main50_verified'] is True
                    and self.receipt.get('identity_unchanged') is True):
                self.receipt.update(candidate_rejected=True,
                    rejection_kind='controlled-admission-rejection',
                    prepare_error_code=PREPARE_ERROR_CODES[outcome - 1])
                raise CandidateUnavailable('controlled prepare admission rejected') from error
            self.receipt.update(candidate_rejected=False, prepare_error_code='unexpected-error')
            raise RuntimeError('unexpected denoise failure: prepare-main50') from error
        try:
            for block, commit in zip(self.blocks, commits):
                commit()
                dispatch = block._h3_region_b_dispatch

                def guarded(this, *args, _dispatch=dispatch):
                    result = None
                    error = None
                    try:
                        if self.context is None:
                            raise RuntimeError('missing forward context')
                        result = _dispatch(*args)
                    except Exception as exc:
                        error = exc
                    # Required before ANY next block/forward attention collective.
                    self._post_agreement(error, 'local-post')
                    return result
                block._h3_region_b_dispatch = types.MethodType(guarded, block)
            if self._identity() != before:
                raise RuntimeError('model identity changed')
            installed = True
        except Exception:
            installed = False
        try:
            self._agree(installed, 'install-main50')
        except BaseException:
            for block, snapshot in zip(self.blocks, snapshots):
                for key in self._attrs:
                    if key in snapshot:
                        setattr(block, key, snapshot[key])
                    else:
                        block.__dict__.pop(key, None)
            raise
        self.identity = before
        self.state = 'ready'
        self.receipt['identity_unchanged'] = True
        self.receipt['shared_python_helpers'] = 1

    def run(self, stage, index, metadata, forward, *args, **kwargs):
        if self.state != 'ready':
            raise RuntimeError('owner faulted/not ready')
        valid = (stage in ('first', 'guide') and type(index) is int and 0 <= index <= 3
                 and type(metadata) is dict and set(metadata) == {'R', 'T', 'local_shape'}
                 and all(type(metadata[k]) is int and metadata[k] > 0 for k in ('R', 'T'))
                 and self.context is None)
        self._agree(valid, 'forward-entry')
        self.context = (stage, index, dict(metadata))
        for b in self.blocks:
            b._h3_local_stage, b._h3_local_forward_index = stage, index
        try:
            result = forward(*args, **kwargs)
        except BaseException:
            # Peer may already be inside native attention: process deadline is
            # mandatory; do NOT issue a mismatched extra status collective here.
            self.state = 'faulted'
            raise
        finally:
            self.context = None
        self.receipt['compile_backend_invocations'] = self.budget.count
        self.receipt['compile_backend_seconds'] = self.budget.seconds
        return result

    def representative_preflight(self, eager, compiled, args, compare):
        """Caller supplies ONE real local GPU input and GPU metrics reducer.

        compare must report device/shape/max/mean/exact metrics (no tolerance
        admission). No automatic quality promotion; no model/tensor cloning.
        Must run collectively before measured full-model work.
        """
        if self.state != 'ready':
            raise RuntimeError('owner not ready')
        error = None
        metrics = None
        try:
            metrics = compare(eager(*args), compiled(*args))
        except Exception as exc:
            error = exc
        self._post_agreement(error, 'representative-preflight')
        self.receipt['representative_metrics'] = metrics
        return metrics


class CandidateRejected(RuntimeError):
    """Measured candidate finding; renderer must not classify as provider failure."""


class CandidateUnavailable(CandidateRejected):
    """Strictly prevalidated local admission unsupported; never GPU acceptance."""


def _controlled(error):
    if isinstance(error, CandidateRejected):
        return True
    cls = type(error)
    if cls.__module__ == 'torch._dynamo.exc':
        if cls.__name__ in ('Unsupported', 'RecompileLimitExceeded', 'CacheLimitExceeded'):
            return True
        if cls.__name__ == 'BackendCompilerFailed':
            cause = getattr(error, 'inner_exception', None)
            return cause is not error and cause is not None and _controlled(cause)
    return False


class TorchStatus:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device

    def min(self, ok, label):
        return bool(self._reduce(int(ok), 'MIN'))

    def rejection_vote(self, code, label):
        return self._reduce(code, 'MAX')

    def _reduce(self, code, operation):
        t = self.torch
        if not t.distributed.is_initialized() or t.distributed.get_world_size() != 2:
            raise RuntimeError('initialized two-rank native group required')
        value = t.tensor(code, dtype=t.int32, device=self.device)
        t.distributed.all_reduce(value, op=getattr(t.distributed.ReduceOp, operation))
        return int(value.item())


def inductor_backend(gm, example_inputs, *, options=None):
    # torch.compile passes custom backend options, not Inductor's registry API.
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs, config_patches=dict(options or {}))


def shape_plan(layout, batch, context):
    """Native logical rows before SP transport padding; no invented text rows."""
    window = context['token'].window
    guide = int(window.guide_video_t) * 112
    reference = int(layout.num_condition_video_rows) - guide
    text = int(layout.text_indices.numel())
    target_video = int(batch.latents.shape[0]) - int(layout.num_condition_video_rows)
    target_audio = int(batch.audio_latents.shape[0]) - int(layout.num_condition_audio_rows)
    guide_audio = int(layout.num_condition_audio_rows)
    if (reference <= 0 or text <= 0 or target_video != 4144 or
            guide not in (0, 784) or guide_audio != 2 * int(window.guide_audio_t)):
        raise ValueError('native continuation packing differs')
    first = reference + text + target_video + target_audio
    # Native continuation guide is seven video times and 37 stereo audio rows.
    lengths = {'first': first, 'guide': first + 784 + 74}
    return reference, text, {k: (1, (v + 1)//2, 5376) for k, v in lengths.items()}


class RequestAdapter:
    """Request hooks are bound early; installation waits for real packed layout."""
    def __init__(self, pipeline, batch, args, policy):
        self.pipeline, self.policy = pipeline, dict(policy)
        self.controller = None
        self.pending = None
        self.closed = False
        self.committed = False

    def on_before_forward(self, stage, batch, index):
        from .fast_continuation_stages import KEY, LAYOUT_KEY
        context, layout = batch.extra[KEY], batch.extra[LAYOUT_KEY]
        self.window_index = context['manifest']['index']
        self.session_id = context['manifest']['session_id']
        if self.closed or self.pending is not None or index != context['steps_held']:
            raise ValueError('denoise hook ordering')
        if self.controller is None:
            self._initialize(stage, batch, context, layout)
        r, t, shapes = shape_plan(layout, batch, context)
        name = 'first' if context['manifest']['index'] == 0 else 'guide'
        self.pending = (name, index, r, t, shapes[name], layout)

    def _initialize(self, stage, batch, context, layout):
        import torch
        status = TorchStatus(torch, batch.latents.device)
        error = None
        try:
            self._preadmit(stage, batch, context, layout, status)
        except Exception as exc:
            error = exc
        if not status.min(error is None, 'request-init'):
            if self.controller is not None:
                self.controller.state = 'faulted'
            raise _AgreementRejected('denoise rank agreement failed: request-init') from error
        if error is not None:
            raise _AgreementRejected('denoise rank agreement failed: request-init') from error
        owner, model = self.controller, self.model
        source, namespace, shapes = self._install_context
        owner.install(source=source, namespace=namespace,
            admission=dict(bf16=True, static_merged=True, fa4=True, portrait=True,
                           resident=True, old_regional=False), stage_shapes=shapes,
            gates=dict(source_pinned=True, real_bf16_weights=True,
                       static_adapter_receipt_verified=True, main50_verified=True))
        snapshots = [(obj, {k: obj.__dict__[k] for k in keys if k in obj.__dict__}, keys)
                     for obj, keys in [(model, ('forward', '_h3_denoise_owner'))] +
                     [(b, ('_h3_local_probe',)) for b in owner.blocks]]
        error = None
        try:
            self._bind_forward(owner, model)
        except Exception as exc:
            error = exc
        try:
            owner._finish_boundary(error, 'bind-forward')
        except BaseException:
            owner.state = 'faulted'
            for obj, saved, keys in snapshots:
                for key in keys:
                    if key in saved:
                        obj.__dict__[key] = saved[key]
                    else:
                        obj.__dict__.pop(key, None)
            raise

    def _preadmit(self, stage, batch, context, layout, status):
        import inspect
        import sys
        import os
        import torch
        if (os.environ.get('H3_REF2VA_ATTENTION_MODE') != 'fa4-packed' or
                os.environ.get('H3_REF2VA_RUNTIME_POLICY') != 'resident-visual-cached' or
                os.environ.get('H3_REF2VA_REGIONAL_COMPILE') == '1' or
                (batch.width, batch.height) != (256, 448)):
            raise ValueError('native FA4 resident portrait policy required')
        from .fast_ref2va import static_merge_receipt
        if static_merge_receipt(self.pipeline) != dict(active=312, merged=312, dynamic=0):
            raise ValueError('static merge receipt differs')
        model = self.pipeline.get_module('transformer')
        if stage.transformer is not model:
            raise ValueError('native stage transformer identity differs')
        if getattr(model, '_h3_denoise_owner', None) is not None:
            raise ValueError('model already owned')
        _, _, shapes = shape_plan(layout, batch, context)
        owner = DenoiseController(model, model.transformer_blocks,
            status=status, compiler=torch.compile,
            backend=inductor_backend, max_compile_count=self.policy['max_backend_compiles'],
            max_compile_seconds=self.policy['max_compile_seconds'])
        # Register before installation so collective failures retain real receipts.
        self.controller, self.model = owner, model
        self.original_forward = model.forward
        namespace = vars(sys.modules[type(model).__module__])
        source = inspect.getsource(sys.modules[type(model).__module__])
        self._install_context = source, namespace, shapes

    def _bind_forward(self, owner, model):
        adapter = self

        def forward(this, hidden_states, audio_hidden_states, encoder_hidden_states,
                    timestep, timestep_indices, token_tags, position_ids,
                    video_indices, audio_indices, text_indices):
            if adapter.pending is None:
                raise RuntimeError('native forward outside hook')
            name, index, r, t, shape, packed = adapter.pending
            logical = int(position_ids.shape[0])
            if ((1, (logical+1)//2, 5376) != shape or
                    logical != int(packed.position_ids.shape[0]) or
                    int(text_indices.numel()) != t):
                raise ValueError('actual native forward metadata drift')
            return owner.run(name, index, dict(R=r, T=t, local_shape=shape),
                adapter.original_forward, hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states, encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, timestep_indices=timestep_indices, token_tags=token_tags,
                position_ids=position_ids, video_indices=video_indices,
                audio_indices=audio_indices, text_indices=text_indices)
        self.wrapped_forward = types.MethodType(forward, model)
        model.forward = self.wrapped_forward
        model._h3_denoise_owner = self
        for block in owner.blocks:
            block._h3_local_probe = self._probe

    def _probe(self, eager, compiled, args):
        owner = self.controller
        if 'representative_metrics' in owner.receipt:
            return
        error = None
        metrics = None
        try:
            raw = self._probe_compute(eager, compiled, args)
            metrics = {k: raw[k] for k in
                       ('exact', 'max', 'mean', 'eager_ms', 'compiled_ms', 'host_seconds')}
            if type(metrics['exact']) is not bool:
                raise RuntimeError('invalid representative exact metric')
            finite = all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k])
                         for k in ('max', 'mean', 'eager_ms', 'compiled_ms', 'host_seconds'))
            if not finite:
                metrics = None
                raise CandidateRejected('representative nonfinite metrics')
            if not (metrics['exact'] is True and
                    0 < metrics['compiled_ms'] < metrics['eager_ms']):
                raise CandidateRejected('representative numerical/benefit gate rejected')
        except Exception as exc:
            error = exc
        owner.receipt['representative_metrics'] = metrics if error is None else None
        try:
            owner._finish_boundary(error, 'representative-probe')
        except BaseException:
            owner.receipt['representative_metrics'] = None
            raise
        owner.receipt['representative_preflight_complete'] = True

    def _probe_compute(self, eager, compiled, args):
        # All potentially throwing local work is inside _probe's capture, before
        # the single common boundary. Fatal CUDA/native hangs need the deadline.
        import torch
        start = time.monotonic()
        expected = eager(*args)
        actual = compiled(*args)
        delta = (expected.float() - actual.float()).abs()
        metrics = dict(exact=bool(torch.equal(expected, actual)),
            max=float(delta.max().item()), mean=float(delta.mean().item()),
            shape=list(actual.shape), device=str(actual.device), dtype=str(actual.dtype),
            interpretation='token intermediate; not per-frame or full-model parity')
        # One warm execution per path; compilation excluded from CUDA timing.
        def measure(fn):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(*args); b.record(); b.synchronize()
            return float(a.elapsed_time(b))
        metrics['eager_ms'], metrics['compiled_ms'] = measure(eager), measure(compiled)
        metrics['host_seconds'] = time.monotonic() - start
        return metrics

    def on_hold_audio(self, stage, batch):
        if self.pending is None:
            raise ValueError('hold without forward')
        self.pending = None

    def on_commit(self, stage, batch):
        if self.pending is not None or batch.extra['_h3_continuation']['steps_held'] != 4:
            raise ValueError('commit before four holds')
        self.committed = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.controller is None:
            return
        # Restore top-level binding; local compiled helpers remain warm on model.
        if hasattr(self, 'original_forward'):
            self.model.forward = self.original_forward
        self.model._h3_denoise_owner = None
        self._write_receipt()

    def _write_receipt(self):
        import json
        import os
        from pathlib import Path
        import torch
        c = self.controller
        rank = torch.distributed.get_rank()
        directory = Path(os.environ['H3_REF2VA_COVERAGE_DIR'])
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        row = dict(rank=rank, session_id=self.session_id, denoise_policy=dict(self.policy),
            window_index=self.window_index, state=c.state, owner_terminal=c.state,
            denoise_installed=c.state == 'ready',
            candidate_rejected=c.receipt.get('candidate_rejected', False),
            parameter_identity_preserved=(hasattr(c, 'identity') and
                c.receipt.get('identity_unchanged', False)),
            representative_preflight_complete=c.receipt.get('representative_preflight_complete', False),
            backend_compiles=c.budget.count, compile_seconds=c.budget.seconds,
            regions=len(c.blocks), coverage_proven=False, trace_available=False,
            extra_status_collectives=c.receipt['agreements'],
            extra_status_seconds=c.receipt['agreement_seconds'],
            representative_metrics=c.receipt.get('representative_metrics'),
            native_pre_attention_failure='external bounded process deadline')
        for key in ('prepare_error_code', 'source_pinned_verified', 'main50_verified',
                    'parameter_dtype_histogram', 'native_type_hashes'):
            if key in c.receipt:
                row[key] = c.receipt[key]
        if 'rejection_kind' in c.receipt:
            row['rejection_kind'] = c.receipt['rejection_kind']
        from .fast_ref2va import utilization_name
        for name in ('controller-' + utilization_name(self.session_id, self.window_index, rank),):
            fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(row, stream, allow_nan=False)


def apply_request(pipeline, batch, args, policy):
    from .utilization_policy import normalize
    normalized = normalize(policy, 'denoise')
    if normalized is None:
        return None
    adapter = getattr(pipeline, '_h3_local_denoise_adapter', None)
    if adapter is None:
        adapter = RequestAdapter(pipeline, batch, args, normalized)
        pipeline._h3_local_denoise_adapter = adapter
    else:
        if (adapter.policy != normalized or not adapter.closed or
                not adapter.committed or adapter.controller.state != 'ready'):
            raise RuntimeError('faulted/drifted denoise request reuse')
        adapter.closed, adapter.committed = False, False
        adapter.model.forward = adapter.wrapped_forward
        adapter.model._h3_denoise_owner = adapter
    return adapter
