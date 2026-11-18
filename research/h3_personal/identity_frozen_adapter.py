"""Experimental training-only frozen Linear branches, not production-enabled.

Lazy torch import; no downloads, quantization, merge or extra PEFT adapter.
Use only around a quiescent training phase: finish backward/checkpoint recompute
before suspend/close. Fresh-base export verification/inference must run without
this session. Provenance/licensing and file-hash validation belong to the caller.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

_A = "_h3_conditioning_a"
_B = "_h3_conditioning_b"
_OWNER = "_h3_conditioning_owner"


@dataclass(frozen=True)
class FrozenProjection:
    target: str
    a: object
    b: object
    scale: float


class FrozenAdapterSession:
    """Attach nonpersistent frozen buffers/hooks without changing subject keys.

    Instances are single-use and single-threaded. Frozen parameters still carry
    derivatives with respect to the input. All paths/tensors are validated
    before any attachment; partial registration failure rolls back this session.
    """
    def __init__(self, model, projections):
        self.model = model
        self.projections = tuple(projections)
        self._handles = []
        self._modules = []
        self._used = False
        self._active = False
        self._enabled = False

    def _prepare(self):
        import torch
        if not self.projections:
            raise ValueError("frozen projections required")
        prepared, seen = [], set()
        for projection in self.projections:
            if (not isinstance(projection, FrozenProjection)
                    or not isinstance(projection.target, str) or not projection.target):
                raise ValueError("explicit projection target required")
            module = self.model.get_submodule(projection.target)
            if hasattr(module, "get_base_layer"):
                if getattr(module, "merged_adapters", []):
                    raise ValueError("merged PEFT layers unsupported")
                module = module.get_base_layer()
            if type(module) is not torch.nn.Linear or module.weight.device.type == "meta":
                raise ValueError("only materialized unquantized Linear supported")
            if module.weight.requires_grad or (module.bias is not None and module.bias.requires_grad):
                raise ValueError("base Linear must be frozen")
            if id(module) in seen or any(hasattr(module, key) for key in (_A, _B, _OWNER)):
                raise ValueError("duplicate/occupied frozen branch target")
            seen.add(id(module))
            a, b = projection.a, projection.b
            if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                    or a.ndim != 2 or b.ndim != 2 or a.shape[0] <= 0
                    or a.shape[1] != module.in_features
                    or b.shape != (module.out_features, a.shape[0])
                    or not a.is_floating_point() or not b.is_floating_point()
                    or a.layout != torch.strided or b.layout != torch.strided
                    or a.device.type == "meta" or b.device.type == "meta"):
                raise ValueError("frozen projection shape/dtype/layout mismatch")
            if (isinstance(projection.scale, bool) or not isinstance(projection.scale, (float, int))
                    or not math.isfinite(projection.scale)):
                raise ValueError("finite explicit scale required")
            if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                raise ValueError("nonfinite frozen projection")
            # Clone after detach: no source gradients or shared mutable storage.
            a = a.detach().to(device=module.weight.device, dtype=module.weight.dtype).clone()
            b = b.detach().to(device=module.weight.device, dtype=module.weight.dtype).clone()
            if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                raise ValueError("frozen projection overflow during dtype conversion")
            prepared.append((module, a, b, float(projection.scale)))
        return prepared

    def __enter__(self):
        if self._used:
            raise RuntimeError("frozen adapter session is single-use")
        self._used = True
        prepared = self._prepare()
        try:
            for module, a, b, scale in prepared:
                setattr(module, _OWNER, self)
                self._modules.append(module)
                module.register_buffer(_A, a, persistent=False)
                module.register_buffer(_B, b, persistent=False)

                def branch(linear, inputs, kwargs, output, scale=scale):
                    if not self._enabled:
                        return output
                    import torch.nn.functional as F
                    x = inputs[0] if inputs else kwargs["input"]
                    delta = F.linear(F.linear(x.to(getattr(linear, _A).dtype),
                                               getattr(linear, _A)), getattr(linear, _B))
                    return output + scale * delta.to(output.dtype)

                self._handles.append(module.register_forward_hook(branch, with_kwargs=True))
            self._active = self._enabled = True
            return self
        except BaseException:
            self.close()
            raise

    @contextmanager
    def suspended(self):
        """Explicitly disable branches; caller must finish backward first."""
        if not self._active:
            raise RuntimeError("frozen adapter is not attached")
        enabled = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            if self._active:
                self._enabled = enabled

    def close(self):
        self._enabled = self._active = False
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for module in reversed(self._modules):
            if getattr(module, _OWNER, None) is self:
                for name in (_A, _B, _OWNER):
                    if hasattr(module, name):
                        delattr(module, name)
        self._modules.clear()

    def __exit__(self, *exc):
        self.close()
        return False


def configure_subject_scope(model, *, start_block=20, end_block=50):
    """Experimental full-key scope setup; call BEFORE creating optimizer/DDP."""
    import torch
    from .identity_checkpoint import _config
    from .identity_training_adapter import subject_parameter_scope
    _config(model)  # preserve the existing sole-default/plain-rank16 contract
    parameters = dict(model.named_parameters())
    names = [name for name in parameters if ".lora_" in name]
    scope = subject_parameter_scope(names, start_block=start_block, end_block=end_block)
    subject = set(names)
    if any(p.requires_grad for name, p in parameters.items() if name not in subject):
        raise ValueError("unexpected trainable non-subject parameter")
    # Validate before mutating flags; a previous trained/resumed adapter cannot
    # silently be relabeled as an identity window with no outside effects.
    for name in scope.frozen:
        p = parameters[name]
        if p.grad is not None or (".lora_B." in name and torch.count_nonzero(p).item()):
            raise ValueError("excluded subject parameters must be pristine with zero B")
    active = set(scope.trainable)
    for name in names:
        parameters[name].requires_grad_(name in active)
    verify_subject_scope(model, scope)
    return scope


def verify_subject_scope(model, scope, *, optimizer=None):
    """Check zero excluded deltas and exact optimizer membership after updates."""
    import torch
    from .identity_checkpoint import _config
    _config(model)
    parameters = dict(model.named_parameters())
    names = {name for name in parameters if ".lora_" in name}
    trainable, frozen = set(scope.trainable), set(scope.frozen)
    if (trainable & frozen or names != trainable | frozen
            or {name for name, p in parameters.items() if p.requires_grad} != trainable):
        raise ValueError("subject scope/gradient flags changed")
    for name in frozen:
        p = parameters[name]
        if p.grad is not None or (".lora_B." in name and torch.count_nonzero(p).item()):
            raise ValueError("excluded subject parameters changed or received gradients")
    if optimizer is not None:
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(parameters[name]) for name in trainable}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("optimizer must contain exactly the active subject parameters")
