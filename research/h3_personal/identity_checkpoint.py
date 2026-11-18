"""Strict, local-only default PEFT adapter artifacts; no training-state resume.

Markers are integrity receipts, not authenticated base provenance or runtime proof.
Heavy dependencies are imported only at the boundary. Injected Boundary is for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

C_GATE_ENABLED = False
TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
SOURCE_PINS = {
    "trainer": "7a1563a267a17da898df3396acbd6a9d24fcb142",
    "h3": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "diffusers": "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
}
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
RECEIPT = "identity_receipt.json"
EXPORTED = "EXPORTED.json"
VERIFIED = "FRESH_BASE_RELOAD_VERIFIED.json"
SCHEMA = "h3-identity-checkpoint-v1"


@dataclass(frozen=True)
class Boundary:
    get: Callable
    set: Callable
    save: Callable
    load: Callable
    torch: Any


def _boundary(boundary=None):
    if boundary is not None:
        return boundary
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import save_file, load_file
    return Boundary(get_peft_model_state_dict, set_peft_model_state_dict,
                    save_file, load_file, torch)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _bytes(value):
    return (json.dumps(_jsonable(value), sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pins(pins):
    """Require source commits and at least one explicit model repo/revision pin."""
    if pins.get("sources") != SOURCE_PINS:
        raise ValueError("source pins differ from audited revisions")
    models = pins.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("explicit model pins required")
    for role, pin in models.items():
        if (not isinstance(role, str) or not isinstance(pin, dict)
                or set(pin) != {"repo_id", "revision"}
                or not isinstance(pin["repo_id"], str) or not pin["repo_id"].strip()
                or not isinstance(pin["revision"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", pin["revision"])):
            raise ValueError("model pins require repo_id and full commit revision")
    _bytes(pins)


def validate_config(config):
    if (config.get("peft_type") != "LORA" or config.get("r") != 16
            or config.get("lora_alpha") != 16 or config.get("lora_dropout") != 0
            or not isinstance(config.get("target_modules"), list)
            or sorted(config["target_modules"]) != sorted(TARGETS)):
        raise ValueError("expected rank16/alpha16/dropout0/four suffix targets")
    for key in ("rank_pattern", "alpha_pattern", "modules_to_save",
                "layers_to_transform", "layers_pattern", "target_parameters",
                "exclude_modules"):
        if config.get(key):
            raise ValueError(f"unsupported adapter option: {key}")
    if config.get("bias", "none") != "none" or any(
        config.get(key, False) for key in ("use_dora", "use_rslora", "lora_bias")
    ):
        raise ValueError("only plain bias-free LoRA supported")


def _config(model):
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"}:
        raise ValueError("exactly one default adapter required")
    active = getattr(model, "active_adapters", None)
    # Diffusers PeftAdapterMixin exposes a method; PEFT models use a property.
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
    if isinstance(active, str):
        active = [active]
    if active != ["default"]:
        raise ValueError("default adapter must be active")
    config = _jsonable(configs["default"].to_dict())
    validate_config(config)
    return config


def _state(model, b):
    return b.get(model, adapter_name="default", save_embedding_layers=False)


def _tensors(state, torch):
    if not isinstance(state, dict) or not state:
        raise ValueError("empty adapter state")
    for key, tensor in state.items():
        if (not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor)
                or tensor.numel() == 0 or tensor.layout != torch.strided
                or not tensor.is_floating_point() or not torch.isfinite(tensor).all().item()):
            raise ValueError(f"invalid/nonfinite adapter tensor: {key}")


def _compare_contract(saved, fresh, torch):
    _tensors(saved, torch)
    _tensors(fresh, torch)
    if saved.keys() != fresh.keys():
        raise ValueError("canonical adapter keys differ (missing or unexpected)")
    for key in fresh:
        if saved[key].shape != fresh[key].shape or saved[key].dtype != fresh[key].dtype:
            raise ValueError(f"shape/dtype mismatch: {key}")


def _equal(saved, actual, torch):
    _compare_contract(saved, actual, torch)
    for key in saved:
        if not torch.equal(saved[key].cpu(), actual[key].detach().cpu()):
            raise ValueError(f"adapter round-trip mismatch: {key}")


def _write(path, data):
    with Path(path).open("xb") as handle:
        handle.write(_bytes(data))
        handle.flush()
        os.fsync(handle.fileno())


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inspect_export(path, *, expected_pins):
    """Stdlib integrity check only. Never equivalent to fresh-base reload."""
    path = Path(path)
    validate_pins(expected_pins)
    marker = json.loads((path / EXPORTED).read_text())
    if (marker.get("schema") != SCHEMA or marker.get("status") != "exported"
            or marker.get("c_gate") is not False
            or set(marker.get("digests", {})) != {WEIGHTS, CONFIG, RECEIPT}):
        raise ValueError("invalid exported marker")
    for name, digest in marker["digests"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"artifact digest mismatch: {name}")
    receipt = json.loads((path / RECEIPT).read_text())
    if (receipt.get("schema") != SCHEMA or receipt.get("pins") != expected_pins
            or receipt.get("adapter") != "default" or receipt.get("c_gate") is not False):
        raise ValueError("receipt mismatch")
    validate_config(json.loads((path / CONFIG).read_text()))
    return marker


def export_adapter(model, destination, *, pins, boundary=None):
    """Publish an immutable new directory by same-filesystem rename.

    Caller must quiesce training; only unsharded, unquantized PEFT is supported.
    No dtype conversion, optimizer, counters, or base weights are saved.
    """
    validate_pins(pins)
    config = _config(model)
    b = _boundary(boundary)
    state = _state(model, b)
    _tensors(state, b.torch)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        b.save(state, str(staging / WEIGHTS))
        with (staging / WEIGHTS).open("rb") as handle:
            os.fsync(handle.fileno())
        _equal(state, b.load(str(staging / WEIGHTS), device="cpu"), b.torch)
        _write(staging / CONFIG, config)
        _write(staging / RECEIPT, {
            "schema": SCHEMA, "adapter": "default", "pins": pins, "c_gate": False,
            "tensor_count": len(state),
            "claims": "integrity only; no authenticated base/runtime/forward/quality proof",
        })
        _write(staging / EXPORTED, {
            "schema": SCHEMA, "status": "exported", "c_gate": False,
            "digests": {name: sha256(staging / name) for name in (WEIGHTS, CONFIG, RECEIPT)},
        })
        inspect_export(staging, expected_pins=pins)
        _sync_dir(staging)
        # Never delete/overwrite an existing checkpoint. Caller serializes publication.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
        _sync_dir(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def strict_reload(fresh_model, path, *, freshly_constructed, expected_pins, boundary=None):
    """Reload ONLY an explicitly fresh caller-built PEFT model, not a resume.

    Freshness is a caller assertion, not something metadata can authenticate.
    Any failure leaves the model disposable and no verified marker. Serialize
    validation/publication with all writers; artifacts must remain immutable.
    """
    path = Path(path)
    verified = path / VERIFIED
    verified.unlink(missing_ok=True)  # invalidate stale success even on early failure
    if freshly_constructed is not True:
        raise ValueError("explicit freshly_constructed=True assertion required")
    exported = inspect_export(path, expected_pins=expected_pins)
    config = json.loads((path / CONFIG).read_text())
    if _config(fresh_model) != config:
        raise ValueError("fresh PEFT config differs from saved actual config")
    b = _boundary(boundary)
    saved = b.load(str(path / WEIGHTS), device="cpu")
    _compare_contract(saved, _state(fresh_model, b), b.torch)
    # Keep an independent baseline even if a boundary mutates its input mapping.
    baseline = {k: v.detach().cpu().clone() for k, v in saved.items()}
    result = b.set(fresh_model, saved, adapter_name="default")
    if getattr(result, "unexpected_keys", ()):
        raise ValueError("PEFT setter reported unexpected keys")
    # Base omissions are expected. Adapter omissions are never acceptable.
    if any("lora_" in key or "modules_to_save" in key
           for key in getattr(result, "missing_keys", ())):
        raise ValueError("PEFT setter reported missing adapter keys")
    _equal(baseline, _state(fresh_model, b), b.torch)
    if inspect_export(path, expected_pins=expected_pins) != exported:
        raise ValueError("artifacts changed during reload")
    marker = {
        "schema": SCHEMA, "status": "fresh-base-reload-verified", "c_gate": False,
        "exported_marker_sha256": sha256(path / EXPORTED),
        "digests": exported["digests"], "freshness": "caller-asserted",
        "claims": "adapter tensor equality only; not authenticated base/runtime/forward/quality",
    }
    temporary = path / ("." + VERIFIED + ".tmp")
    try:
        _write(temporary, marker)
        os.replace(temporary, verified)
        _sync_dir(path)
    except BaseException:
        verified.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def inspect_verified(path, *, expected_pins):
    """Future uploader prerequisite, NOT an authenticated validation attestation."""
    exported = inspect_export(path, expected_pins=expected_pins)
    path = Path(path)
    marker = json.loads((path / VERIFIED).read_text())
    if (marker.get("schema") != SCHEMA
            or marker.get("status") != "fresh-base-reload-verified"
            or marker.get("c_gate") is not False
            or marker.get("freshness") != "caller-asserted"
            or marker.get("digests") != exported["digests"]
            or marker.get("exported_marker_sha256") != sha256(path / EXPORTED)):
        raise ValueError("invalid fresh-base reload marker")
    return marker
