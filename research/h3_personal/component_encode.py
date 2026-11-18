"""Executable, injectable Ref2VA component workers; no implicit model loading.

Inputs are already-trimmed clips, one image per sample, batch one. See
``docs/h3-component-encode.md``. Importing this module needs only stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from . import component_cache as cache

C_GATE_ENABLED = False
PLAN_SCHEMA = "h3-component-encode-v1"
RECIPES = {
    "video": "decode_video:24fps/cover-fit;target:posterior-mode/normalize/patch2",
    "audio": "extract_audio:32k-stereo/800-grid;recover-real-PCM-if-None;posterior-mode/channel-major",
    "reference": "RGB/no-exif-transpose;resolve_reference_image_size/prepare_reference_image;seed42/fp16-roundtrip/normalize/patch2",
    "text": "encode_ref2va_prompt:ordered-one-image/labelled-block/no-chat/layer50/bf16",
    "identity_uncond": "encode_ref2va_prompt:single-space/same-image/layer50/bf16/paired-v2",
}


def digest(value):
    return hashlib.sha256(cache.canonical(value)).hexdigest()


def _source(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("source must be a regular file")
    return {"path": str(path), "sha256": cache.sha256(path)}


def _check_sources(unit):
    for source in unit["sources"].values():
        if cache.sha256(source["path"]) != source["sha256"]:
            raise ValueError("source bytes changed after planning")


def make_plan(samples, encoder_sha256, *, attested_by, media_shards=2):
    """Hash local inputs and deduplicate *components*, not whole samples.

    encoder_sha256 maps all five kinds to explicit code/config/weight identities.
    This does not authenticate those caller-supplied identities. No weights read.
    """
    if type(media_shards) is not int or media_shards not in (1, 2):
        raise ValueError("media_shards must be 1 or 2")
    if (set(encoder_sha256) != cache.KINDS
            or not all(cache.digest_ok(v) for v in encoder_sha256.values())
            or not isinstance(attested_by, str) or not attested_by.strip()):
        raise ValueError("explicit encoder hashes and attestor required")
    units, records, ids = {}, [], set()
    source_cache = {}

    def source(path):
        key = str(Path(path).resolve(strict=True))
        if key not in source_cache:
            source_cache[key] = _source(key)
        return source_cache[key]

    def unit(kind, sources, settings, dependencies=None):
        # Paths and sample IDs never enter reusable identities. Byte hashes do.
        recipe = cache.canonical({"engine": PLAN_SCHEMA, "aviad_revision": cache.REVISION,
                                  "kind": kind, "operation": RECIPES[kind],
                                  "settings": settings, "dependencies": dependencies or {}}).decode()
        source_hash = digest({k: v["sha256"] for k, v in sources.items()})
        att = {"aviad_revision": cache.REVISION, "recipe": recipe,
               "source_sha256": source_hash, "encoder_sha256": encoder_sha256[kind],
               "attested_by": attested_by}
        # Attestor is evidence, not a numerical recipe; a new producer can reuse.
        key = digest({k: v for k, v in att.items() if k != "attested_by"})
        units.setdefault(key, {"kind": kind, "sources": sources, "settings": settings,
                               "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
                               "dependencies": dependencies or {}, "attestation": att})
        return key

    allowed = {"id", "video", "audio", "reference_image", "caption", "width", "height", "num_frames", "quality_pool"}
    for row in samples:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("unsupported sample fields (use already-trimmed clips)")
        sid = row.get("id")
        if (not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sid)
                or sid in (".", "..") or sid in ids):
            raise ValueError("invalid/duplicate sample ID")
        ids.add(sid)
        geometry = {k: row[k] for k in ("width", "height", "num_frames")}
        vr, ar = cache.geometry_rows(geometry["width"], geometry["height"], geometry["num_frames"])
        if not isinstance(row.get("caption"), str) or not isinstance(row.get("reference_image"), (str, Path)):
            raise ValueError("caption and exactly one reference image required")
        video, audio, image = source(row["video"]), source(row.get("audio") or row["video"]), source(row["reference_image"])
        if 'quality_pool' in row and row['quality_pool'] not in ('identity','motion'):
            raise ValueError('quality_pool must preserve the reviewed identity/motion decision')
        v = unit("video", {"video": video}, geometry)
        a = unit("audio", {"audio": audio}, {"num_frames": geometry["num_frames"]})
        r = unit("reference", {"image": image}, {})
        t = unit("text", {"image": image, "caption": {"path": None, "sha256": hashlib.sha256(row["caption"].encode()).hexdigest()}},
                 {"prompt": row["caption"]}, {"reference": r})
        # Caption is an inline source, not a file; workers separately verify it.
        p = unit("identity_uncond", {"image": image}, {"prompt": " "}, {"reference": r, "conditional": t})
        records.append({"id": sid, **geometry, "video_rows": vr, "audio_rows": ar,
                        "has_audio": True, "caption": row["caption"], "source": video["path"],
                        "units": dict(video=v, audio=a, reference=r, text=t, identity_uncond=p),
                        **({'quality_pool':row['quality_pool']} if 'quality_pool' in row else {})})
    if not records:
        raise ValueError("empty samples")
    media = sorted(k for k, v in units.items() if v["kind"] in ("video", "audio"))
    return {"schema": PLAN_SCHEMA, "c_gate_enabled": False, "units": units, "samples": records,
            "media_shards": [media[i::media_shards] for i in range(media_shards)]}


def _verify_unit(unit):
    if hashlib.sha256(unit["attestation"]["recipe"].encode()).hexdigest() != unit["recipe_sha256"]:
        raise ValueError("recipe hash mismatch")
    if digest({k: v["sha256"] for k, v in unit["sources"].items()}) != unit["attestation"]["source_sha256"]:
        raise ValueError("source manifest hash mismatch")
    files = {k: v for k, v in unit["sources"].items() if v["path"] is not None}
    _check_sources({"sources": files})
    if "caption" in unit["sources"]:
        if hashlib.sha256(unit["settings"]["prompt"].encode()).hexdigest() != unit["sources"]["caption"]["sha256"]:
            raise ValueError("inline caption hash mismatch")


def recover_audio(source, num_frames):
    """Decode actual PCM when pinned extract_audio discarded pure silence.

    A missing/undecodable/empty track is a hard failure, NOT a zero placeholder.
    Pinned align_waveform's short-track padding is retained after real decode.
    """
    import torch
    from h3_trainer.preprocessing.media import align_waveform, read_wav
    with tempfile.TemporaryDirectory(prefix="h3-real-audio-") as tmp:
        wav = Path(tmp) / "audio.wav"
        result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
                                 "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "32000",
                                 "-c:a", "pcm_s16le", str(wav)], capture_output=True, timeout=300)
        if result.returncode or not wav.exists():
            raise ValueError("required real audio cannot be decoded: " + result.stderr.decode(errors="replace")[:200])
        waveform = read_wav(wav)
        if waveform.ndim != 2 or waveform.shape[0] != 2 or not waveform.shape[1] or not bool(torch.isfinite(waveform).all()):
            raise ValueError("required audio is empty/invalid")
        return align_waveform(waveform, num_frames)


class PinnedBackend:
    """Wrap ALREADY loaded H3Encoders. Never constructs or downloads models.

    Media workers need video/audio VAEs; common reference stage needs video VAE;
    common text stage needs conditioner. They can use separate backend instances.
    """
    def __init__(self, encoders, source_root):
        from research.install_identity_guidance import PINS
        from h3_trainer.preprocessing import encoders as encoder_module, media
        from h3_trainer import constants, model_loader, packing
        root = Path(source_root).resolve()
        for module in (encoder_module, media, packing, constants, model_loader):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(root))
            if relative not in PINS or cache.sha256(path) != PINS[relative]:
                raise ValueError("loaded preprocessing source does not match pinned Aviad")
        if not isinstance(encoders, encoder_module.H3Encoders):
            raise TypeError("requires actual H3Encoders")
        self.encoders, self.media, self.packing = encoders, media, packing

    def video(self, unit):
        clip = self.media.decode_video(unit["sources"]["video"]["path"], **unit["settings"])
        return {"latents": self.encoders.encode_video_target(clip.frames)}

    def audio(self, unit):
        path, frames = unit["sources"]["audio"]["path"], unit["settings"]["num_frames"]
        waveform = self.media.extract_audio(path, frames)
        if waveform is None:
            waveform = recover_audio(path, frames)
        return {"latents": self.encoders.encode_audio(waveform)}

    def image(self, unit):
        from PIL import Image
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import prepare_reference_image, resolve_reference_image_size
        with Image.open(unit["sources"]["image"]["path"]) as original:
            original = original.convert("RGB")
            width, height = resolve_reference_image_size(original.width, original.height)
            return prepare_reference_image(original, height, width)

    def reference(self, unit, image):
        encoded = self.encoders.encode_reference("image", image=image)
        if encoded.audio_rows is not None or encoded.video_rows is None:
            raise ValueError("requires image-only reference")
        return {"latents": encoded.video_rows, "geometry": encoded.geometry[None]}

    def presentation(self, image, geometry):
        import torch
        refs = self.packing.prepared_references_from_cache([("image", None, torch.tensor(geometry, dtype=torch.int64))])
        refs[0].image = image
        return refs

    def text(self, prompt, references):
        embeds, tags = self.encoders.encode_ref2va_prompt(prompt, references)
        return {"prompt_embeds": embeds, "text_token_tags": tags}


class Engine:
    """Workers produce exclusive receipts; only finish() publishes index.json.

    root must be private to this plan. Existing validated v2 components may be
    relocated into root and supplied as reuse_catalog. No legacy certification.
    """
    def __init__(self, root, plan, backend, *, reuse_catalog=None):
        self.root, self.plan, self.backend = Path(root), plan, backend
        if plan.get("schema") != PLAN_SCHEMA or plan.get("c_gate_enabled") is not False:
            raise ValueError("unsupported plan/gate")
        self.root.mkdir(parents=True, exist_ok=True)
        self.reuse = reuse_catalog or {}
        if self.reuse:
            cache.validate_catalog(self.root, self.reuse)
        self.bindings = {}
        for uid, unit in plan["units"].items():
            att = unit["attestation"]
            matches = [(key, entry) for key, entry in self.reuse.items() if entry["kind"] == unit["kind"]
                       and all(entry["attestation"][k] == att[k] for k in att if k != "attested_by")]
            if len(matches) > 1:
                raise ValueError("ambiguous reusable recipe: multiple artifacts")
            if matches:
                self.bindings[uid] = matches[0]
        shards = plan["media_shards"]
        flat = [uid for shard in shards for uid in shard]
        expected = {k for k, u in plan["units"].items() if u["kind"] in ("video", "audio")}
        if len(shards) not in (1, 2) or len(flat) != len(set(flat)) or set(flat) != expected:
            raise ValueError("each unique media unit must be assigned exactly once")

    def _receipt(self, name, bindings):
        payload = {"plan_sha256": digest(self.plan), "bindings": {u: {"key": k, "entry": e} for u, (k, e) in bindings.items()}}
        with (self.root / (name + ".json")).open("x") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def _read(self, name):
        payload = json.loads((self.root / (name + ".json")).read_text())
        if payload["plan_sha256"] != digest(self.plan):
            raise ValueError("receipt belongs to another plan")
        return {u: (v["key"], v["entry"]) for u, v in payload["bindings"].items()}

    def _write(self, uid, tensors, metadata=None, deps=None):
        from safetensors.torch import save_file
        unit = self.plan["units"][uid]
        kind = unit["kind"]
        if kind == "video":
            spec = unit["settings"]
        elif kind == "reference":
            spec = {"geometry": tensors["geometry"][0].tolist()}
        else:
            spec = {"rows": int(tensors["latents" if kind == "audio" else "prompt_embeds"].shape[0])}
        if kind == "identity_uncond":
            spec.update(deps)
        objects = self.root / "objects"
        objects.mkdir(exist_ok=True)
        target = objects / (uid + ".safetensors")
        # Same-filesystem exclusive link: no partial tensors visible to other workers.
        fd, tmp = tempfile.mkstemp(prefix=".encode-", dir=objects)
        os.close(fd)
        try:
            save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, tmp, metadata=metadata)
            os.link(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
        entry = {"kind": kind, "path": str(target.relative_to(self.root)), "sha256": cache.sha256(target),
                 "spec": spec, "attestation": unit["attestation"]}
        return cache.component_id(entry), entry

    def run_media(self, rank):
        if type(rank) is not int or not 0 <= rank < len(self.plan["media_shards"]):
            raise ValueError("invalid media rank")
        result = {}
        for uid in self.plan["media_shards"][rank]:
            unit = self.plan["units"][uid]
            _verify_unit(unit)
            if uid in self.bindings:
                result[uid] = self.bindings[uid]
            else:
                tensors = getattr(self.backend, unit["kind"])(unit)
                _verify_unit(unit)
                result[uid] = self._write(uid, tensors)
        cache.validate_catalog(self.root, {k: e for k, e in result.values()}) if result else None
        return self._receipt(f"media-{rank}", result)

    def run_references(self):
        result = {}
        for uid, unit in self.plan["units"].items():
            if unit["kind"] != "reference":
                continue
            _verify_unit(unit)
            if uid in self.bindings:
                result[uid] = self.bindings[uid]
            else:
                image = self.backend.image(unit)
                tensors = self.backend.reference(unit, image)
                _verify_unit(unit)
                result[uid] = self._write(uid, tensors)
        cache.validate_catalog(self.root, {k: e for k, e in result.values()})
        return self._receipt("references", result)

    def run_text(self):
        references = self._read("references")
        result = dict(references)
        # One prepared image/descriptor reused for BOTH branches and all captions
        # of this reference. Do not retain the corpus of prepared pixels in RAM.
        for rid, (_, ref) in references.items():
            runit = self.plan["units"][rid]
            _verify_unit(runit)
            image = self.backend.image(runit)
            presentation = self.backend.presentation(image, ref["spec"]["geometry"])
            shared_probe = None
            for kind in ("text", "identity_uncond"):
                for uid, unit in self.plan["units"].items():
                    if unit["kind"] != kind or unit["dependencies"].get("reference") != rid:
                        continue
                    _verify_unit(unit)
                    if uid in self.bindings:
                        result[uid] = self.bindings[uid]
                        continue
                    deps, metadata = None, None
                    if kind == "identity_uncond":
                        cond_key, cond = result[unit["dependencies"]["conditional"]]
                        deps = {"conditional": cond_key, "reference": result[rid][0]}
                        metadata = {"schema": cache.PROBE_SCHEMA, "aviad_revision": cache.REVISION,
                                    "probe_prompt": " ", "conditional_sha256": cond["sha256"],
                                    "reference_sha256": ref["sha256"]}
                    if kind == "identity_uncond" and shared_probe is not None:
                        tensors = shared_probe
                    else:
                        tensors = self.backend.text(unit["settings"]["prompt"], presentation)
                        if kind == "identity_uncond":
                            shared_probe = tensors
                    _verify_unit(unit)
                    result[uid] = self._write(uid, tensors, metadata, deps)
            _verify_unit(runit)
        cache.validate_catalog(self.root, {k: e for k, e in result.values()})
        return self._receipt("text", {u: v for u, v in result.items() if u not in references})

    def finish(self):
        bindings = {}
        stages = {f"media-{i}": set(shard) for i, shard in enumerate(self.plan["media_shards"])}
        stages["references"] = {u for u, v in self.plan["units"].items() if v["kind"] == "reference"}
        stages["text"] = {u for u, v in self.plan["units"].items() if v["kind"] in ("text", "identity_uncond")}
        for name, expected in stages.items():
            receipt = self._read(name)
            if set(receipt) != expected:
                raise ValueError("incomplete worker outputs or wrong stage assignment")
            for uid, pair in receipt.items():
                if uid in bindings:
                    raise ValueError("duplicate worker assignment")
                unit = self.plan["units"][uid]
                entry = pair[1]
                if pair[0] != cache.component_id(entry) or entry["kind"] != unit["kind"] or any(
                        entry["attestation"][k] != unit["attestation"][k] for k in unit["attestation"] if k != "attested_by"):
                    raise ValueError("receipt recipe/address mismatch")
                bindings[uid] = pair
        if set(bindings) != set(self.plan["units"]):
            raise ValueError("incomplete worker outputs")
        catalog = {k: e for k, e in bindings.values()}
        samples = []
        for row in self.plan["samples"]:
            sample = {k: v for k, v in row.items() if k != "units"}
            sample["components"] = {kind: bindings[uid][0] for kind, uid in row["units"].items()}
            sample["text_rows"] = catalog[sample["components"]["text"]]["spec"]["rows"]
            sample["reference_geometry"] = [catalog[sample["components"]["reference"]]["spec"]["geometry"]]
            samples.append(sample)
        return cache.write_index(self.root, catalog, samples)


def main(argv=None):
    """Merge-only CLI. Encoding is explicit Python API with loaded encoders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    print(Engine(args.root, plan, None).finish())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
