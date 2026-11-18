"""Validated shared-component cache for the pinned Ref2VA guidance subset.

Stdlib at import time; no encoder execution and no gate-C enablement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REVISION = "7a1563a267a17da898df3396acbd6a9d24fcb142"
SCHEMA = "h3-component-index-v2"
PROBE_SCHEMA = "h3-component-probe-v2"
C_GATE_ENABLED = False
KINDS = {"video", "audio", "reference", "text", "identity_uncond"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def component_id(component):
    # Location is deliberately excluded: identical verified bytes can be relocated.
    return hashlib.sha256(canonical({k: v for k, v in component.items() if k != "path"})).hexdigest()


def contained_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be relative")
    if ".." in Path(relative).parts:
        raise ValueError("artifact traversal")
    root = Path(root).resolve()
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact must be a contained regular file")
    return path


def provenance_status(component):
    """Legacy bytes/shape alone do not establish how an encoder produced them."""
    att = component.get("attestation")
    if not isinstance(att, dict):
        return "unresolved"
    if (set(att) != {"aviad_revision", "recipe", "source_sha256", "encoder_sha256", "attested_by"}
            or att["aviad_revision"] != REVISION
            or not all(isinstance(att[k], str) and att[k].strip() for k in ("recipe", "attested_by"))
            or not all(digest_ok(att[k]) for k in ("source_sha256", "encoder_sha256"))):
        return "unresolved"
    return "attested"  # caller assertion, NOT independent verification of encoder execution


def positive(value):
    return type(value) is int and value > 0


def geometry_rows(width, height, frames):
    if (not all(positive(x) for x in (width, height, frames)) or width % 32 or height % 32
            or frames < 22 or frames % 17 != 5 or frames > 360):
        raise ValueError("invalid pinned target geometry")
    return ((frames - 5) // 17 * 5 + 2) * (height // 32) * (width // 32), 2 * round(frames / 24 * 40)


def validate_catalog(root, catalog):
    """Hash and inspect actual tensors, never merely declared tensor manifests."""
    import torch
    from safetensors.torch import load_file
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("empty component catalog")
    loaded = {}
    for key, entry in catalog.items():
        if (not isinstance(entry, dict) or set(entry) != {"kind", "path", "sha256", "spec", "attestation"}
                or entry["kind"] not in KINDS or not digest_ok(entry["sha256"])
                or key != component_id(entry)):
            raise ValueError("invalid component address/descriptor")
        if provenance_status(entry) != "attested":
            raise ValueError("unresolved component provenance: explicit attestation required")
        path = contained_path(root, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("artifact hash mismatch")
        tensors = load_file(str(path), device="cpu")
        spec, kind = entry["spec"], entry["kind"]
        if not isinstance(spec, dict):
            raise ValueError("invalid spec")
        expected = {"latents"}
        if kind == "reference":
            expected.add("geometry")
        elif kind in {"text", "identity_uncond"}:
            expected = {"prompt_embeds", "text_token_tags"}
        if set(tensors) != expected:
            raise ValueError("incompatible tensor keys")
        if kind == "video":
            if set(spec) != {"width", "height", "num_frames"}:
                raise ValueError("invalid video spec")
            rows, _ = geometry_rows(spec["width"], spec["height"], spec["num_frames"])
            shape, dtype, name = (rows, 96), torch.float32, "latents"
        elif kind == "audio":
            if set(spec) != {"rows"} or not positive(spec["rows"]):
                raise ValueError("invalid audio spec")
            shape, dtype, name = (spec["rows"], 32), torch.float32, "latents"
        elif kind == "reference":
            if set(spec) != {"geometry"}:
                raise ValueError("invalid reference spec")
            g = spec["geometry"]
            if (not isinstance(g, list) or len(g) != 5 or any(type(v) is not int for v in g)
                    or g[0] != 0 or g[1] != 1 or g[4] != 0
                    or not positive(g[2]) or not positive(g[3]) or g[2] % 2 or g[3] % 2):
                raise ValueError("requires one image reference with even latent spatial grid")
            actual = tensors["geometry"]
            if actual.dtype != torch.int64 or actual.shape != (1, 5) or actual.tolist() != [g]:
                raise ValueError("reference geometry mismatch")
            shape, dtype, name = (g[2] // 2 * (g[3] // 2), 96), torch.float32, "latents"
        else:
            fields = {"rows"} if kind == "text" else {"rows", "conditional", "reference"}
            if set(spec) != fields or not positive(spec["rows"]):
                raise ValueError("invalid prompt spec")
            shape, dtype, name = (spec["rows"], 5120), torch.bfloat16, "prompt_embeds"
            tags = tensors["text_token_tags"]
            if (tags.dtype != torch.int64 or tags.shape != (spec["rows"],)
                    or not bool(((tags == 0) | (tags == 1)).all())
                    or not bool((tags == 0).any()) or not bool((tags == 1).any())):
                raise ValueError("invalid reference-bearing prompt tags")
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("tensor shape/dtype/finite mismatch")
        # Validate sequentially: never retain an entire multi-GB latent dataset.
        loaded[key] = {"vision_rows": int((tensors["text_token_tags"] == 0).sum())
                       if kind in {"text", "identity_uncond"} else 0}
        del tensors, tensor
    from safetensors import safe_open
    for key, entry in catalog.items():
        if entry["kind"] != "identity_uncond":
            continue
        spec = entry["spec"]
        cond, ref = catalog.get(spec["conditional"]), catalog.get(spec["reference"])
        if not cond or not ref or cond["kind"] != "text" or ref["kind"] != "reference":
            raise ValueError("probe dependencies missing/incompatible")
        expected = {"schema": PROBE_SCHEMA, "aviad_revision": REVISION, "probe_prompt": " ",
                    "conditional_sha256": cond["sha256"], "reference_sha256": ref["sha256"]}
        with safe_open(str(contained_path(root, entry["path"])), framework="pt", device="cpu") as handle:
            if handle.metadata() != expected:
                raise ValueError("probe paired metadata mismatch (sample IDs are not shared provenance)")
        if loaded[key]["vision_rows"] != loaded[spec["conditional"]]["vision_rows"]:
            raise ValueError("probe reference vision tags mismatch")
    return loaded


def validate_index(root, payload):
    if (payload.get("version") != 2 or payload.get("schema") != SCHEMA
            or payload.get("c_gate_enabled") is not False or payload.get("batch_size") != 1):
        raise ValueError("unsupported component index/gate/batch")
    catalog = payload["components"]
    validate_catalog(root, catalog)
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples or payload.get("num_samples") != len(samples):
        raise ValueError("empty/incomplete sample index")
    ids = set()
    used = set()
    for sample in samples:
        sid = sample.get("id")
        if (not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sid)
                or sid in (".", "..") or sid in ids):
            raise ValueError("invalid/duplicate sample ID")
        ids.add(sid)
        refs = sample.get("components")
        if not isinstance(refs, dict) or set(refs) != KINDS:
            raise ValueError("sample requires actual video/audio/text/reference/paired probe")
        for kind, key in refs.items():
            if key not in catalog or catalog[key]["kind"] != kind:
                raise ValueError("sample component missing/wrong kind")
            used.add(key)
        video = catalog[refs["video"]]["spec"]
        vr, ar = geometry_rows(sample["width"], sample["height"], sample["num_frames"])
        if (video != {k: sample[k] for k in video} or sample.get("video_rows") != vr
                or sample.get("audio_rows") != ar or sample.get("has_audio") is not True
                or catalog[refs["audio"]]["spec"]["rows"] != ar
                or sample.get("text_rows") != catalog[refs["text"]]["spec"]["rows"]
                or sample.get("reference_geometry") != [catalog[refs["reference"]]["spec"]["geometry"]]):
            raise ValueError("sample geometry/row counts/real audio mismatch")
        probe = catalog[refs["identity_uncond"]]["spec"]
        if probe["conditional"] != refs["text"] or probe["reference"] != refs["reference"]:
            raise ValueError("sample probe pairing mismatch")
    if used != set(catalog):
        raise ValueError("catalog must contain exactly the selected complete samples' components")
    return payload


def write_index(root, catalog, samples):
    """Publish a fresh index only, never merge legacy index records or overwrite."""
    root = Path(root)
    payload = {"version": 2, "schema": SCHEMA, "c_gate_enabled": False, "batch_size": 1,
               "components": catalog, "samples": samples, "num_samples": len(samples)}
    validate_index(root, payload)
    with (root / "index.json").open("x") as handle:
        handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return root / "index.json"


def load_component(dataset, source, sample_id, required):
    from safetensors.torch import load_file
    record = dataset.component_samples[sample_id]
    key = record["components"].get(source)
    if key is None:
        if required:
            raise ValueError("required component missing")
        return None
    entry = dataset.component_index["components"][key]
    path = contained_path(dataset.root, entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("component changed after validation")
    return load_file(str(path), device="cpu")


def load_probe(dataset, record):
    payload = load_component(dataset, "identity_uncond", record.id, True)
    return {"identity_uncond_text": payload["prompt_embeds"],
            "identity_uncond_tags": payload["text_token_tags"]}
