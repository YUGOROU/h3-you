"""Default-off, instance-only native main-block region-B experiment.

No model copy, archive mutation, global compiler/precision settings, or fallback.
CPU contract tests are not BF16 GPU parity or speed acceptance. Caller admission
is an assertion about an already resident, static-merged BF16 FA4 portrait model.
Only install on the 50 main blocks, never reference/conditioning blocks.
Runtime ownership and collective boundaries live in denoise_execution.py.
"""
import ast
import hashlib
import types

NATIVE_SHA256 = 'b7d470816af7b18620710e3b40c7625f2658d1950d7859f018ecc5130d379753'
_ARGS = ('self', 'hidden_states', 'attention_output', 'adaln_indices',
         'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp',
         'use_modulate_fusion', 'use_exact_modulate')


class AdmissionUnsupported(ValueError):
    """Fixed local guard evidence, not a generic ValueError classification."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _safe_fault(exc):
    # Fixed vocabulary: even custom exception class names may contain secrets.
    for cls in (ValueError, TypeError, RuntimeError, AssertionError, MemoryError):
        if type(exc) is cls:
            return cls.__name__
    return 'Exception'


def extract(source):
    """Validate exact native bytes before compiling any extracted Python code."""
    if hashlib.sha256(source.encode()).hexdigest() != NATIVE_SHA256:
        raise ValueError('native source SHA mismatch')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'MiniMaxH3TransformerBlock')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    boundary = next(i for i, n in enumerate(forward.body)
                    if isinstance(n, ast.Delete) and ast.unparse(n) == 'del norm_hidden_states') + 1
    region = forward.body[boundary:]
    helper = ast.FunctionDef(name='_h3_region_b',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in _ARGS],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=region, decorator_list=[])
    # The entire original eager branch remains byte-for-byte in the input and
    # AST-identical in this replacement; only an explicit flag adds a return.
    branch = ast.parse('if getattr(self, "_h3_local_post_attention", False):\n'
                       '    return self._h3_region_b_dispatch(' + ', '.join(_ARGS[1:]) + ')').body[0]
    forward.body = forward.body[:boundary] + [branch] + region
    forward.decorator_list = []
    forward.returns = None
    for arg in forward.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[helper, forward], type_ignores=[]))
    return compile(module, '<h3-native-region-b>', 'exec')


def install(block, *, source, namespace, enabled=False, main_block_index=None,
            admission=None, stage_shapes=None, compile_factory=None,
            prepare_only=False, prepared_source=None):
    """Install only after explicit admission; returns False without work by default.

    namespace must be the actual native module globals. stage_shapes maps exactly
    first/guide to actual local (1,L,5376) shapes. Set block._h3_local_stage before
    each forward, plus _h3_local_forward_index (strict int 0..3). Each stage
    admits one initial and one later tensor/flag signature, at most four total.
    No eviction or compiler retry. A faulted receipt forbids model promotion.
    Compilation is lazy at the first admitted post-attention call; errors escape.
    """
    if enabled is False:
        return False
    if enabled is not True or type(main_block_index) is not int or not 0 <= main_block_index < 50:
        raise ValueError('explicit main50 block admission required')
    required = dict(bf16=True, static_merged=True, fa4=True, portrait=True,
                    resident=True, old_regional=False)
    if (type(admission) is not dict or admission.keys() != required.keys()
            or any(type(admission[k]) is not bool or admission[k] is not v
                   for k, v in required.items())):
        raise ValueError('BF16 static merged FA4 portrait resident admission required')
    if type(stage_shapes) is not dict or set(stage_shapes) != {'first', 'guide'}:
        raise ValueError('two logical stage shapes required')
    shapes = dict(stage_shapes)
    for shape in shapes.values():
        if (type(shape) is not tuple or len(shape) != 3
                or any(type(d) is not int for d in shape) or shape[0] != 1
                or shape[2] != 5376 or type(shape[1]) is not int or shape[1] <= 0):
            raise ValueError('expected actual local (1,L,5376) shape')
    if getattr(block.ff, 'chunk_tokens', 0) or getattr(block.ff, 'use_mxfp8', False):
        raise AdmissionUnsupported('ffn-configuration-unsupported',
                                   'FFN chunking/quantization is not admitted')
    if hasattr(block, '_h3_region_b_dispatch'):
        raise ValueError('already installed')
    if compile_factory is not None and not callable(compile_factory):
        raise ValueError('callable compiler required')
    if hashlib.sha256(source.encode()).hexdigest() != NATIVE_SHA256:
        raise ValueError('native source SHA mismatch')
    code = extract(source) if prepared_source is None else prepared_source[0]
    native_class = namespace.get('MiniMaxH3TransformerBlock')
    native_forward = getattr(native_class, 'forward', None)
    if (type(native_class) is not type or type(block) is not native_class
            or not isinstance(native_forward, types.FunctionType)
            or native_forward.__globals__ is not namespace
            or getattr(block.forward, '__func__', None) is not native_forward):
        raise AdmissionUnsupported('native-provenance-unsupported',
                                   'native class/forward globals provenance required')
    # Compare executable native forward semantics to the pinned source.
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'MiniMaxH3TransformerBlock')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    verification = dict(namespace)
    # Nested generator code constants inherit future flags from the native
    # module. Recompile under the same flags, not this helper module's flags.
    import __future__
    actual = native_forward.__code__
    future_mask = 0
    for feature in __future__.all_feature_names:
        future_mask |= getattr(__future__, feature).compiler_flag
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<native-verification>', 'exec', flags=actual.co_flags & future_mask,
                 dont_inherit=True), verification)
    expected = verification['forward'].__code__
    if (actual.co_code != expected.co_code or actual.co_consts != expected.co_consts
            or actual.co_names != expected.co_names):
        raise AdmissionUnsupported('native-provenance-unsupported',
                                   'native forward source provenance mismatch')
    if prepared_source is None:
        scope = dict(namespace)
        exec(code, scope)
        helper, forward = scope['_h3_region_b'], scope['forward']
    else:
        _, helper, forward = prepared_source
    compiled = None
    signatures = {}
    receipt = dict(state='installed', fault=None, signatures=signatures,
                   compile_settings=dict(fullgraph=True, dynamic=False,
                                         options={'emulate_precision_casts': True,
                                                  'triton.cudagraphs': False}),
                   precision_policy='unchanged; no TF32/global precision mutation',
                   native_sha256=NATIVE_SHA256)

    def dispatch(self, *args):
        nonlocal compiled
        if receipt['state'] == 'faulted':
            raise RuntimeError('local compile block permanently faulted: ' + receipt['fault'])
        try:
            return admitted_dispatch(self, *args)
        except Exception as exc:
            receipt['state'] = 'faulted'
            receipt['fault'] = _safe_fault(exc)
            raise

    def admitted_dispatch(self, *args):
        nonlocal compiled
        stage = getattr(self, '_h3_local_stage', None)
        index = getattr(self, '_h3_local_forward_index', None)
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError('explicit forward index 0..3 required')
        phase = 'initial' if index == 0 else 'later'
        key = (stage, phase)
        if type(stage) is not str or stage not in shapes or tuple(args[0].shape) != shapes[stage]:
            raise ValueError('unadmitted stage/local shape')
        if getattr(self.ff, 'chunk_tokens', 0) or getattr(self.ff, 'use_mxfp8', False):
            raise ValueError('FFN configuration drift')
        tensors = args[:7]
        if str(tensors[0].dtype) != 'torch.bfloat16':
            raise ValueError('BF16 activation required')
        signature = tuple((tuple(t.shape), str(t.dtype), str(t.device), tuple(t.stride()))
                          for t in tensors) + tuple(args[7:])
        if key in signatures and signatures[key] != signature:
            raise ValueError('static stage/phase signature drift')
        signatures[key] = signature
        try:
            if compiled is None:
                factory = compile_factory
                if factory is None:
                    import torch
                    factory = torch.compile
                compiled = factory(helper, **receipt['compile_settings'])
                if not callable(compiled):
                    raise TypeError('compiler returned non-callable')
            probe = getattr(self, '_h3_local_probe', None)
            if probe is not None:
                probe(helper, compiled, (self, *args))
            result = compiled(self, *args)
        except Exception as exc:
            receipt['state'] = 'faulted'
            receipt['fault'] = _safe_fault(exc)
            raise
        receipt['state'] = 'active'
        return result

    # All validation/source compilation precedes instance mutation. No modules or
    # parameters are replaced, registered, copied, or rebound.
    def commit():
        block._h3_local_compile_receipt = receipt
        block._h3_region_b_dispatch = types.MethodType(dispatch, block)
        block.forward = types.MethodType(forward, block)
        block._h3_local_post_attention = True
        return True
    if prepare_only:
        return commit
    return commit()
