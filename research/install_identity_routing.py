#!/usr/bin/env python3
"""Stage opt-in Trainer/TREAD source overlay; never alter installed Diffusers.

Output is a review/build plan, not a runnable dependency install or GPU launch.
The deployment builder must explicitly install BOTH exact overlays later.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from research import install_identity_sampling as sampling

DIFFUSERS_SHA256 = "45f1d4a4e9e72128f0d6c5b5d66379b61131a354b58d2776280a83f6c89f8d51"


def patch_transformer(source):
    if hashlib.sha256(source.encode()).hexdigest() != DIFFUSERS_SHA256:
        raise ValueError("Diffusers H3 source differs from pinned pristine bytes")
    substitute = sampling.components.guidance.substitute
    source = substitute(source, "        return_dict: bool = True,\n",
                        "        return_dict: bool = True,\n        identity_routing_plan=None,\n")
    source = substitute(source, "        for block in self.transformer_blocks:\n",
        "        routed_state = None\n"
        "        original_metadata = (adaln_indices, rotary_emb, attention_mask)\n"
        "        if identity_routing_plan is not None:\n"
        "            if not torch.is_grad_enabled():\n"
        "                raise ValueError('Routing forbidden during inference/no-grad')\n"
        "            if not 0 <= identity_routing_plan.start_block < identity_routing_plan.end_block <= len(self.transformer_blocks):\n"
        "                raise ValueError('Routing block window exceeds model depth')\n"
        "        for block_index, block in enumerate(self.transformer_blocks):\n"
        "            if identity_routing_plan is not None and block_index == identity_routing_plan.start_block:\n"
        "                from h3_trainer.identity_routing_tensors import route_packed_activations\n"
        "                routed_state = route_packed_activations(hidden_states, rotary_emb, adaln_indices, attention_mask, identity_routing_plan)\n"
        "                hidden_states, rotary_emb = routed_state.hidden_states, routed_state.rotary_emb\n"
        "                adaln_indices, attention_mask = routed_state.adaln_indices, routed_state.attention_mask\n")
    source = substitute(source,
        "                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)\n",
        "                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)\n"
        "            if identity_routing_plan is not None and block_index + 1 == identity_routing_plan.end_block:\n"
        "                hidden_states = routed_state.rejoin(hidden_states)\n"
        "                adaln_indices, rotary_emb, attention_mask = original_metadata\n"
        "                routed_state = None\n")
    ast.parse(source)
    return source


def plan(source_root, diffusers_source):
    sources = sampling.plan(source_root)
    substitute = sampling.components.guidance.substitute
    config = "src/h3_trainer/config.py"
    sources[config] = substitute(sources[config], "class FlexibleStrategyConfig(ConfigBaseModel):",
        "class FlexibleStrategyConfig(ConfigBaseModel):\n"
        "    identity_tread_ratio: float = Field(default=0.0, ge=0.0, lt=1.0, strict=True)\n"
        "    identity_tread_start: int = Field(default=2, ge=0, strict=True)\n"
        "    identity_tread_end: int = Field(default=47, le=50, strict=True)\n")
    trainer = "src/h3_trainer/trainer.py"
    sources[trainer] = substitute(sources[trainer], "from h3_trainer import logger",
        "from h3_trainer import logger\n"
        "from h3_trainer.identity_routing_forward import routing_forward, validate_routing_config")
    sources[trainer] = substitute(sources[trainer], "        validate_training_config(config)",
        "        validate_training_config(config)\n        validate_routing_config(config)")
    sources[trainer] = substitute(sources[trainer], "        return self.model(**packed.to_model_kwargs())",
        "        strategy = self.config.training_strategy\n"
        "        return routing_forward(self.model, packed, ratio=strategy.identity_tread_ratio,\n"
        "            seed=self.config.seed + self.state.step, start_block=strategy.identity_tread_start,\n"
        "            end_block=strategy.identity_tread_end)")
    for name in ("identity_routing", "identity_routing_tensors", "identity_routing_forward"):
        sources[f"src/h3_trainer/{name}.py"] = (Path(__file__).parent / f"h3_personal/{name}.py").read_text()
    result = {"trainer/" + name: text for name, text in sources.items()}
    result["diffusers/src/diffusers/models/transformers/transformer_minimax_h3.py"] = patch_transformer(
        Path(diffusers_source).read_text())
    for name, text in result.items():
        ast.parse(text, filename=name)
    return result


def stage(source_root, diffusers_source, destination):
    """Atomically publish reviewable source trees + byte-bound receipt.

    Deliberately does not install/import the overlay or modify worker defaults.
    """
    sources = plan(source_root, diffusers_source)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "h3-identity-routing-overlay-v1",
        "trainer_revision": sampling.components.guidance.REVISION,
        "diffusers_pristine_sha256": DIFFUSERS_SHA256,
        "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(sources.items())},
        "installed": False, "production_enabled": False,
        "contains_weights": False, "contains_credentials": False,
        "actual_h3_gpu_verified": False,
    }
    temporary = Path(tempfile.mkdtemp(prefix=".h3-routing-overlay-", dir=destination.parent))
    try:
        for name, text in sources.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (temporary / "OVERLAY_MANIFEST.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def inspect_stage(destination):
    """Offline exact file inventory/hash check; no runtime/quality assertion."""
    destination = Path(destination)
    receipt = json.loads((destination / "OVERLAY_MANIFEST.json").read_text())
    if (receipt.get("schema") != "h3-identity-routing-overlay-v1"
            or receipt.get("trainer_revision") != sampling.components.guidance.REVISION
            or receipt.get("diffusers_pristine_sha256") != DIFFUSERS_SHA256
            or any(receipt.get(key) is not False for key in
                   ("installed", "production_enabled", "contains_weights", "contains_credentials", "actual_h3_gpu_verified"))):
        raise ValueError("invalid source-only overlay receipt")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("overlay files required")
    actual = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    if actual != set(files) | {"OVERLAY_MANIFEST.json"}:
        raise ValueError("overlay inventory mismatch")
    for name, digest in files.items():
        path = destination / name
        if (path.is_symlink() or not path.resolve().is_relative_to(destination.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError("overlay path/hash mismatch")
    return receipt
