#!/usr/bin/env python3
"""Bounded local-input production worker. No provisioning, downloads, or retries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

FROZEN_SHA256 = "17900bcdcbc5a6e2441b95b824d5d7778fdfd11a461bdbc3b4dc7f63ceacd634"
DATASET_SHA256 = "77e49d18d116be9717f8cd0558f2c1da765c3bb6db1fd27d9393d522fc22b0df"
MODEL_PIN = "42ed227ee7df40d41602854ae760620d6eb651fe"
# Include EU outermost regions/Åland, not just member-state ISO codes.
EXCLUDED = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE AX GF GP MQ RE YT MF GB KR US".split())
ISO_COUNTRIES = set((
    "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ "
    "CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR "
    "GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP "
    "KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ "
    "NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ "
    "TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW"
).split())
CANONICAL = ("adapter_model.safetensors", "adapter_config.json", "identity_receipt.json",
             "EXPORTED.json", "FRESH_BASE_RELOAD_VERIFIED.json")


def helper():
    try:
        return importlib.import_module("research.h3_personal.identity_checkpoint")
    except ModuleNotFoundError:
        return importlib.import_module("h3_personal.identity_checkpoint")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def report(path, value):
    """One bounded, atomic snapshot, never an unbounded event log."""
    data = json.dumps(value, indent=2, allow_nan=False).encode()
    if len(data) > 32768:
        raise ValueError("report exceeds 32KiB")
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def finite_number(value, name, low, high):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"invalid {name}")
    return value


def territory_gate(approval):
    """Explicit reviewed compute evidence, not a hosting/collection URL inference."""
    country = approval.get("compute_country")
    if approval.get("evidence_mode") == "operator_hf_acceptance":
        # External user's contract interpretation, not authenticated rights-holder proof.
        if ("compute_country" not in approval or country is not None
                or approval.get("provider") != "huggingface_jobs"
                or approval.get("approved") is not True
                or not isinstance(approval.get("approved_by"), str)
                or not approval["approved_by"].strip()
                or approval.get("risk_acknowledged") is not True
                or approval.get("evidence_kind") != "operator_hf_acceptance"
                or approval.get("license_allows_execution_country") is True):
            raise ValueError("explicit HF operator acceptance with unknown country required")
        path = Path(approval.get("evidence_file", ""))
        if not path.is_file() or sha256(path) != approval.get("evidence_sha256"):
            raise ValueError("operator evidence digest mismatch")
        return
    if (not isinstance(country, str) or country not in ISO_COUNTRIES or country in EXCLUDED):
        raise ValueError("approved non-excluded ISO compute country required")
    if (approval.get("approved") is not True or not approval.get("approved_by")
            or approval.get("evidence_kind") not in
            {"provider_compute_country_statement", "rights_holder_execution_authorization"}):
        raise ValueError("explicit reviewed territory evidence required")
    path = Path(approval["evidence_file"])
    if not path.is_file() or sha256(path) != approval.get("evidence_sha256"):
        raise ValueError("territory evidence digest mismatch")
    # Approval is supplied by the external launcher/operator, not authenticated here.


def upload_run_id(r):
    """Stable namespace: explicit safe ID, or the entire resolved work-root chain."""
    if "run_id" in r:
        value = r["run_id"]
        if (not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None):
            raise ValueError("safe run_id required (1-128 ASCII letters/digits/_/-)")
        return value
    work = Path(r["work"])
    if not work.is_absolute() or work.resolve() == Path("/"):
        raise ValueError("absolute non-root work path required for derived run_id")
    return "work-" + hashlib.sha256(str(work.resolve()).encode()).hexdigest()


class BoundedTrainingStop(Exception):
    """Only raised by the post-update production logger at the wall reserve."""


def handoff_helper():
    try:
        return importlib.import_module('research.h3_personal.identity_handoff')
    except ModuleNotFoundError:
        return importlib.import_module('h3_personal.identity_handoff')


def handoff_sampling_start(r):
    if 'warm_start' not in r:
        return 0
    warm = r['warm_start']
    if sha256(warm['metadata']) != warm['metadata_sha256']:
        raise ValueError('handoff metadata changed')
    metadata = read_json(warm['metadata'])
    cursor = metadata.get('sampling_cursor')
    if cursor is None:
        return 0  # legacy weight-only handoff makes no sampler-continuity claim
    handoff_helper().validate_sampling_cursor(cursor, minimum_draws=metadata['parent_step'])
    if (cursor['mode'] != r.get('sampling_mode', 'replacement') or cursor['seed'] != 42
            or cursor['epoch_budget'] != 715 or cursor['index_sha256'] != r['index_sha256']):
        raise ValueError('handoff sampler recipe/index mismatch')
    return cursor['consumed_draws']


def training_recipe_digest(r):
    payload = config_payload(r)
    recipe = {
        'pins': r['pins'], 'dataset': DATASET_SHA256, 'index': r['index_sha256'],
        'lora': payload['lora'], 'strategy': payload['training_strategy'],
        'optimization': {k: v for k, v in payload['optimization'].items() if k != 'steps'},
        'acceleration': payload['acceleration'], 'sampling': payload['data']['identity_sampling_mode'],
        'seed': payload['seed'], 'routing_overlay': r.get('routing_overlay_sha256'),
        'conditioning_adapter': r.get('conditioning_adapter_sha256'),
    }
    if 'runtime_overlay' in r:
        recipe['runtime_overlay'] = {key: r['runtime_overlay'][key] for key in
                                     ('stage_sha256', 'base_inventory_sha256')}
    return handoff_helper().recipe_digest(recipe)


def validate_request(r):
    if r.get("schema") != "h3-identity-train-request-v1":
        raise ValueError("request schema mismatch")
    if type(r.get("max_updates")) is not int or r["max_updates"] < 1:
        raise ValueError("explicit positive max_updates required")
    if 'operator_managed_stages' in r and (type(r['operator_managed_stages']) is not bool or
            (r['operator_managed_stages'] and r.get('preview_only') is not True)):
        raise ValueError('operator-managed stages require inference-only request')
    finite_number(r.get("overall_seconds"), "overall_seconds", 1, 21600)
    finite_number(r.get("overall_started_at"), "overall_started_at", 1, time.time() + 5)
    from research.modal_gpu_plan import authorized_rate_limit
    rate = finite_number(r.get("rate_usd_hour"), "rate_usd_hour", 0.001, authorized_rate_limit(r))
    from research.h3_personal.execution_budget import cap
    cumulative_cap=cap(r.get('hf_price_precision_approval'))
    spent = finite_number(r.get("spent_and_reserved_usd"), "spent_and_reserved_usd", 0, cumulative_cap)
    if spent + rate * r["overall_seconds"] / 3600 > cumulative_cap:
        raise ValueError("cumulative budget cap exceeded")
    upload_run_id(r)
    if 'runtime_overlay' in r:
        overlay = r['runtime_overlay']
        if (not isinstance(overlay, dict) or set(overlay) !=
                {'stage', 'stage_sha256', 'base_diffusers', 'base_inventory_sha256'}
                or any(not isinstance(overlay[k], str) or not Path(overlay[k]).is_absolute()
                       for k in ('stage', 'base_diffusers'))
                or any(not isinstance(overlay[k], str) or re.fullmatch('[0-9a-f]{64}', overlay[k]) is None
                       for k in ('stage_sha256', 'base_inventory_sha256'))):
            raise ValueError('literal independently pinned runtime overlay required')
    if type(r.get('checkpoint_escrow', False)) is not bool:
        raise ValueError('checkpoint_escrow must be boolean')
    if 'warm_start' in r:
        warm = r['warm_start']
        if not isinstance(warm, dict) or set(warm) != {'checkpoint', 'metadata', 'metadata_sha256'}:
            raise ValueError('explicit weight-only checkpoint/metadata pin required')
        if sha256(warm['metadata']) != warm['metadata_sha256']:
            raise ValueError('handoff metadata hash mismatch')
        handoff_helper().validate_handoff(warm['checkpoint'], read_json(warm['metadata']),
            pins=r['pins'], expected_recipe_sha256=training_recipe_digest(r))
    territory_gate(r["territory_approval"])
    pins = r["pins"]
    helper().validate_pins(pins)
    if pins["models"] != {"transformer_ref": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": MODEL_PIN}}:
        raise ValueError("actual checkpoint pipeline H3 pin required")
    if sha256(r["frozen_selection"]) != FROZEN_SHA256:
        raise ValueError("not the fixed715 selection bytes")
    if sha256(Path(r["cache"]) / "index.json") != r["index_sha256"]:
        raise ValueError("index digest mismatch")
    if r.get("initial_checks", 3) not in (1, 2, 3):
        raise ValueError("initial checks must be 1-3 updates inside production")
    if type(r.get("checkpoint_interval", 25)) is not int or r.get("checkpoint_interval", 25) < 1:
        raise ValueError("positive checkpoint interval required")
    finite_number(r.get("learning_rate", 1e-5), "learning_rate", 1e-12, 1)
    if type(r.get("max_seq_tokens", 70000)) is not int or r.get("max_seq_tokens", 70000) < 1024:
        raise ValueError("invalid token bound")
    for key in ("RANK", "LOCAL_RANK"):
        if int(os.environ.get(key, "0")) != 0:
            raise ValueError("single rank only")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single process only")
    return r


def validate_weights(r):
    """Offline inventory checked before loading. Pin attribution is caller-supplied."""
    root = Path(r["model"]).resolve()
    inventory = r["model_inventory"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if not inventory or actual != set(inventory):
        raise ValueError("complete local model inventory required")
    for name, digest in inventory.items():
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or sha256(path) != digest):
            raise ValueError("model inventory mismatch")


def config_payload(r):
    payload = {
        "model": {"model_path": r["model"], "variant": "ref2va", "training_mode": "lora",
                  "load_checkpoint": None},
        "lora": {"rank": 16, "alpha": 16, "dropout": 0,
                 "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]},
        "training_strategy": {
            "name": "flexible", "identity_guidance_scale": 4.0, "identity_guidance_sigma_min": 0.15,
            "video": {"is_generated": True, "latents_dir": "latents", "conditions": [
                {"type": "reference", "modality": "image", "probability": 1.0}]},
            "audio": {"is_generated": False, "latents_dir": "audio_latents", "conditions": []}},
        "optimization": {"steps": r["max_updates"], "learning_rate": r.get("learning_rate", 1e-5),
                         "batch_size": 1, "gradient_accumulation_steps": 1, "optimizer_type": "adamw",
                         "max_seq_tokens": r.get("max_seq_tokens", 70000)},
        "acceleration": {"strategy": "ddp", "quantization": "none", "mixed_precision_mode": "bf16"},
        "data": {"preprocessed_data_root": r["cache"], "val_split_every": 0,
                 "num_dataloader_workers": 0, "shuffle": True, "identity_sampling_budget": 715,
                 "identity_sampling_mode": r.get('sampling_mode', 'replacement')},
        "validation": {"sample_media": False, "samples": [], "interval": None,
                       "skip_initial_validation": True},
        "checkpoints": {"interval": r.get("checkpoint_interval", 25), "save_training_state": "off",
                        "identity_peft_export": True},
        "hub": {"push_to_hub": False}, "wandb": {"enabled": False},
        "seed": 42, "output_dir": str(Path(r["work"]) / "runs"),
    }
    start_draw = handoff_sampling_start(r)
    if start_draw:
        payload['data']['identity_sampling_start_draw'] = start_draw
    return payload


def materialize_runtime(r):
    """One exclusive source install; baseline remains the default."""
    runtime = Path(r['work']) / 'runtime'
    if 'runtime_overlay' not in r:
        try:
            installer = importlib.import_module('research.install_identity_checkpoint')
        except ModuleNotFoundError:
            installer = importlib.import_module('install_identity_checkpoint')
        return installer.install(r['source'], runtime)
    from research import apply_identity_runtime_overlay as installer
    installer.install(**r['runtime_overlay'], destination=runtime)
    report(Path(r['work']) / 'runtime-overlay-pin.json',
                {'sha256': sha256(runtime / 'RUNTIME_OVERLAY.json')})
    return runtime


def load_runtime(r, *, training=False):
    runtime = Path(r["work"]) / "runtime"
    if 'runtime_overlay' in r:
        from research import apply_identity_runtime_overlay as overlay
        pin = read_json(Path(r['work']) / 'runtime-overlay-pin.json')['sha256']
        if sha256(runtime / 'RUNTIME_OVERLAY.json') != pin:
            raise ValueError('installed overlay receipt changed')
        overlay.inspect(runtime, stage_sha256=r['runtime_overlay']['stage_sha256'])
        shadow = runtime / 'diffusers-shadow'
        if not training:
            imported = sys.modules.get('diffusers')
            if (any(Path(entry).resolve().is_relative_to(shadow.resolve()) for entry in sys.path)
                    or (imported is not None and getattr(imported, '__file__', None)
                        and Path(imported.__file__).resolve().is_relative_to(shadow.resolve()))):
                raise ValueError('verification requires a fresh pristine Diffusers process')
        if training:
            if any(name == 'diffusers' or name.startswith('diffusers.') for name in sys.modules):
                raise ValueError('routing shadow requires a fresh import process')
            sys.path.insert(0, str(shadow))
            import diffusers
            if not Path(diffusers.__file__).resolve().is_relative_to(shadow.resolve()):
                raise ValueError('unexpected Diffusers routing implementation')
        # Verification does NOT import the Diffusers shadow: fresh pristine base.
    # Never reuse an earlier patched tree. Only supervisor invokes install.
    sys.path.insert(0, str(runtime / "src"))
    import h3_trainer
    if not Path(h3_trainer.__file__).resolve().is_relative_to(runtime.resolve()):
        raise ValueError("unexpected trainer import")
    from h3_trainer.config import H3TrainerConfig
    return H3TrainerConfig.model_validate(config_payload(r))


def verify_dataset(r, config):
    from h3_trainer.datasets import PrecomputedDataset
    from h3_trainer.identity_sampling import IdentityBatchSampler
    dataset = PrecomputedDataset(config.data.preprocessed_data_root,
                                 config.training_strategy.data_sources(), split="train", val_split_every=0,
                                 max_seq_tokens=config.optimization.max_seq_tokens)
    expected = read_json(r["frozen_selection"])["samples"]
    actual = [(rec.id, (rec.extras or {}).get("quality_pool")) for rec in dataset.records]
    wanted = [(rec["id"], rec["quality_pool"]) for rec in expected]
    if (dataset.component_index is None or actual != wanted
            or Counter(pool for _, pool in actual) != {"identity": 590, "motion": 125}):
        raise ValueError("actual loader must retain all fixed715 ordered IDs/pools")
    IdentityBatchSampler(dataset, sample_budget=715, max_seq_tokens=config.optimization.max_seq_tokens,
                         mode=config.data.identity_sampling_mode)
    # Execute the real patched loader for EVERY input; metadata-only validation is insufficient.
    for index in range(len(dataset)):
        dataset[index]
    return dataset


def check_gradients(model, torch):
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
        raise ValueError("missing or nonfinite backward gradients")


def checked_loss(original, torch):
    def compute(video, audio, packed):
        for prediction, target, mask in (
                (video, packed.clean_video, packed.video_loss_mask),
                (audio, packed.clean_audio, packed.audio_loss_mask)):
            # External clean audio is conditioning, not a generated/loss target.
            # Its zero-row loss mask must not be compared against clean audio rows.
            if mask is not None and not mask.any().item():
                continue
            if prediction is None:
                raise ValueError("missing generated forward prediction")
            selected = prediction[..., mask, :] if mask is not None else prediction
            if selected.shape != target.shape:
                raise ValueError("forward target-row prediction shape mismatch")
            if not torch.isfinite(prediction).all().item():
                raise ValueError("nonfinite forward prediction")
        loss = original(video, audio, packed)
        if loss.total.numel() != 1 or not torch.isfinite(loss.total).all().item():
            raise ValueError("nonfinite/nonscalar loss")
        return loss
    return compute


def verify_initial_adapter(r, config, checkpoint):
    """Actual fresh H3 base on CPU; avoids a second base competing for GPU VRAM."""
    import gc
    import torch
    from h3_trainer.model_loader import load_transformer
    from h3_trainer.lora import set_trainable
    # Preserve the running training RNG while constructing the verification model.
    with torch.random.fork_rng(devices=[]):
        fresh = load_transformer(config.model.model_path, variant='ref2va', dtype=torch.bfloat16,
                                 device=torch.device('cpu'), quantization='none')
        try:
            set_trainable(fresh, 'lora', config.lora)
            helper().strict_reload(fresh, checkpoint, freshly_constructed=True, expected_pins=r['pins'])
        finally:
            del fresh
            gc.collect()


def train_stage(r):
    if r.get('operator_managed_stages'):
        raise ValueError('operator-managed stages cannot train')
    import torch
    config = load_runtime(r, training=True)
    from h3_trainer.trainer import H3Trainer
    from h3_trainer.checkpointing import checkpoint_dir
    validate_weights(r)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible GPU required")
    verify_dataset(r, config)
    checkpoints = helper()
    snapshot = Path(r["work"]) / "training-snapshot.json"

    class ProductionTrainer(H3Trainer):
        def save(self, step, *, force_escrow=False):
            destination = checkpoint_dir(self.output_dir, step) / "identity-peft"
            if not destination.exists():
                super().save(step)  # Canonical export inside actual atomic checkpoint staging.
            checkpoints.inspect_export(destination, expected_pins=r["pins"])
            report(snapshot, {"status": "exported_only", "step": step,
                              "checkpoint": str(destination), "quality_verified": False})
            if (r.get('checkpoint_escrow', False) and r.get('output_repo')
                    and (force_escrow or step % r.get('checkpoint_interval', 25) == 0
                         or step == min(r.get('initial_checks', 3), r['max_updates']))):
                if self.state.samples_seen != step:
                    raise ValueError('singleton handoff draw/update counters differ')
                upload_escrow(r, destination, step, sampling_cursor={
                    'consumed_draws': self.sampler.start_draw + self.state.samples_seen,
                    'mode': self.sampler.mode, 'epoch_budget': self.sampler.sample_budget,
                    'seed': self.sampler.seed, 'index_sha256': r['index_sha256']})

    trainer = ProductionTrainer(config)  # Actual pinned constructor, new TrainingState/optimizer.
    trainer.setup()
    if 'warm_start' in r:
        if trainer.state.step != 0 or trainer.state.samples_seen != 0 or trainer.optimizer.state:
            raise ValueError('weight-only handoff requires new counters/optimizer')
        warm = r['warm_start']
        # Strict reload publishes a local verification marker: never mutate the
        # pinned read-only source checkpoint fetched from another provider/run.
        import shutil
        local_checkpoint = Path(r['work']) / 'warm-start-input'
        local_checkpoint.mkdir(mode=0o700, exist_ok=False)
        for name in handoff_helper().ARTIFACTS:
            shutil.copyfile(Path(warm['checkpoint']) / name, local_checkpoint / name)
        lineage = handoff_helper().restore_weights(trainer._core_model(), local_checkpoint,
            read_json(warm['metadata']), pins=r['pins'], expected_recipe_sha256=training_recipe_digest(r))
        if trainer.sampler.start_draw != handoff_sampling_start(r):
            raise ValueError('actual sampler did not restore the bound cursor')
        lineage['sampling_cursor_restored'] = read_json(warm['metadata']).get('sampling_cursor') is not None
        report(Path(r['work']) / 'weight-handoff.json', lineage)
    from peft import get_peft_model_state_dict
    initial_adapter = {k: v.detach().cpu().clone() for k, v in
                       get_peft_model_state_dict(trainer._core_model(), adapter_name='default',
                                                 save_embedding_layers=False).items()}
    trainer.strategy.compute_loss = checked_loss(trainer.strategy.compute_loss, torch)
    original_step = trainer.optimizer.step
    updates = [0]

    def step(*args, **kwargs):
        check_gradients(trainer._core_model(), torch)
        result = original_step(*args, **kwargs)
        updates[0] += 1
        return result

    original_log = trainer.run_logger.log
    deadline = r["overall_started_at"] + r["overall_seconds"]
    last_update_at = time.monotonic()
    longest_update_seconds = 0.0

    def log(metrics, step_number, **kwargs):
        nonlocal last_update_at, longest_update_seconds
        if not (1 <= updates[0] <= r["max_updates"]
                and trainer.state.step == updates[0] == step_number):
            raise ValueError("post-update state/logger/optimizer count mismatch")
        now = time.monotonic()
        longest_update_seconds = max(longest_update_seconds, now - last_update_at)
        last_update_at = now
        for value in metrics.values():
            if type(value) in (float, int) and not math.isfinite(value):
                raise ValueError("nonfinite update metrics")
        original_log(metrics, step_number, **kwargs)
        print('TRAIN_UPDATE', json.dumps({'step': step_number, 'max_updates': r['max_updates'],
              'longest_update_seconds': round(longest_update_seconds, 3),
              'metrics': {k: metrics[k] for k in ('loss', 'learning_rate', 'grad_norm')
                          if k in metrics and type(metrics[k]) in (float, int)}}), flush=True)
        # Reserve five minutes or two observed update times for export and
        # external verification/upload. This does not replace the hard watchdog.
        reserve = max(300.0, 2 * longest_update_seconds)
        if step_number < r["max_updates"] and time.time() >= deadline - reserve:
            raise BoundedTrainingStop()
        if step_number <= min(r.get("initial_checks", 3), r["max_updates"]):
            # After backward/optimizer and state increment, same run, no reset.
            trainer.save(step_number)
            if step_number == min(r.get('initial_checks', 3), r['max_updates']):
                actual = get_peft_model_state_dict(trainer._core_model(), adapter_name='default',
                                                  save_embedding_layers=False)
                if actual.keys() != initial_adapter.keys() or not any(
                        not torch.equal(initial_adapter[k], v.detach().cpu()) for k, v in actual.items()):
                    raise ValueError('optimizer did not change the canonical adapter')
                initial_adapter.clear()
                destination = checkpoint_dir(trainer.output_dir, step_number) / 'identity-peft'
                verify_initial_adapter(r, config, destination)
                # Publish this verified early checkpoint even if later training fails.
                if r.get('output_repo'):
                    from huggingface_hub import constants as hub_constants
                    offline = hub_constants.HF_HUB_OFFLINE
                    try:
                        hub_constants.HF_HUB_OFFLINE = False
                        upload_validated(r, destination)
                    finally:
                        hub_constants.HF_HUB_OFFLINE = offline

    trainer.run_logger.log = log
    trainer.optimizer.step = step
    wall_stop = False
    try:
        trainer.train()  # Actual patched guidance forward + pinned backward/optimizer loop.
    except BoundedTrainingStop:
        wall_stop = True
    if not (1 <= updates[0] <= r["max_updates"] and trainer.state.step == updates[0]):
        raise ValueError("actual optimizer update/state count mismatch")
    if not wall_stop and updates[0] != r["max_updates"]:
        raise ValueError("requested finite update count not achieved")
    # Wall stop bypasses upstream's final save: export actual state, not the cap.
    # Optimizer state remains configured off.
    trainer.save(trainer.state.step, force_escrow=True)
    report(snapshot, {"status": "exported_only", "step": trainer.state.step,
                      "actual_updates": updates[0], "max_updates": r["max_updates"],
                      "training_outcome": "bounded_stop" if wall_stop else "max_updates_complete",
                      "samples_seen": trainer.state.samples_seen, "epoch": trainer.state.epoch,
                      "checkpoint": str(checkpoint_dir(trainer.output_dir, trainer.state.step) / "identity-peft"),
                      "quality_verified": False})
    # Process exits before verification; CUDA model/optimizer are not retained.


def verify_stage(r, checkpoint, forward_id=None):
    import torch
    config = load_runtime(r)
    validate_weights(r)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible GPU required")
    from h3_trainer.model_loader import load_transformer
    from h3_trainer.lora import set_trainable
    model = load_transformer(config.model.model_path, variant="ref2va", dtype=torch.bfloat16,
                             device=torch.device("cuda:0"), quantization="none")
    set_trainable(model, "lora", config.lora)
    checkpoints = helper()
    checkpoints.strict_reload(model, checkpoint, freshly_constructed=True, expected_pins=r["pins"])
    if forward_id:
        from h3_trainer.flow_matching import SigmaPair
        from h3_trainer.training_strategies import get_training_strategy
        from h3_trainer.identity_guidance import training_forward
        dataset = verify_dataset(r, config)
        ids = [rec.id for rec in dataset.records]
        if forward_id not in ids:
            raise ValueError("fixed forward ID outside frozen dataset")
        strategy = get_training_strategy(config.training_strategy)
        model.eval()
        with torch.no_grad():
            packed = strategy.prepare(dataset[ids.index(forward_id)],
                                      SigmaPair.from_u(0.5, video_shift=config.flow_matching.video_shift,
                                                       audio_shift=config.flow_matching.audio_shift),
                                      torch.device("cuda:0"), torch.bfloat16, noise_seed=42)
            video, audio = training_forward(lambda p: model(**p.to_model_kwargs()), packed,
                                             config.training_strategy)
            checked_loss(strategy.compute_loss, torch)(video, audio, packed)
        report(Path(r["work"]) / "fixed-forward.json", {
            "status": "finite_fixed_sample_guidance_forward", "sample_id": forward_id,
            "u": 0.5, "seed": 42, "checkpoint_export_sha256": sha256(Path(checkpoint) / "EXPORTED.json"),
            "claims": "finite forward/loss only; not generation, likeness, sync, or real H3 quality proof"})
    checkpoints.inspect_verified(checkpoint, expected_pins=r["pins"])


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Kill descendants too, even if the session leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def bounded_process(command, *, deadline, env=None):
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("overall preprocessing+training+verification deadline exhausted")
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
        if code:
            raise RuntimeError(f"stage exited {code}")
    except subprocess.TimeoutExpired as exc:
        kill_group(process)
        raise TimeoutError("overall deadline: process group stopped") from exc
    except BaseException:
        kill_group(process)
        raise


def upload_escrow(r, checkpoint, step, *, sampling_cursor=None):
    """Single private atomic commit under existing supervisor deadline; no retries.

    Exported-only escrow is NOT the verified/final adapter upload namespace.
    """
    from huggingface_hub import HfApi, CommitOperationAdd, constants as hub_constants
    metadata = handoff_helper().escrow_metadata(checkpoint, pins=r['pins'],
        recipe_sha256=training_recipe_digest(r), run_id=upload_run_id(r), step=step,
        sampling_cursor=sampling_cursor)
    checkpoint_name = Path(checkpoint).parent.name
    match = re.fullmatch(r'checkpoint-([0-9]+)', checkpoint_name)
    if match is None or int(match.group(1)) != step:
        raise ValueError('canonical checkpoint directory/update binding required')
    prefix = f"identity/{upload_run_id(r)}/escrow/{checkpoint_name}"
    latest = Path(r['work']) / 'latest-escrow.json'
    if latest.is_file():
        previous = read_json(latest)
        if (previous.get('parent_step') == step and previous.get('parent_run_id') == metadata['parent_run_id']
                and previous.get('recipe_sha256') == metadata['recipe_sha256']
                and previous.get('artifacts') == metadata['artifacts']
                and previous.get('sampling_cursor') == metadata['sampling_cursor']):
            return  # exact same local checkpoint already committed; no duplicate upload
    offline = hub_constants.HF_HUB_OFFLINE
    try:
        hub_constants.HF_HUB_OFFLINE = False
        api = HfApi()
        if api.repo_info(r['output_repo']).private is not True:
            raise ValueError('escrow repository must already be private')
        operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{name}',
            path_or_fileobj=str(Path(checkpoint) / name)) for name in handoff_helper().ARTIFACTS]
        operations.append(CommitOperationAdd(path_in_repo=f'{prefix}/HANDOFF.json',
            path_or_fileobj=json.dumps(metadata, sort_keys=True, allow_nan=False).encode()))
        result = api.create_commit(repo_id=r['output_repo'], operations=operations,
            commit_message='Private exported-only weight handoff; no optimizer/quality claim')
        report(Path(r['work']) / 'latest-escrow.json', dict(metadata,
            repo_id=r['output_repo'], revision=result.oid, prefix=prefix))
    finally:
        hub_constants.HF_HUB_OFFLINE = offline


def upload_validated(r, checkpoint):
    """Only final strict-verified canonical artifacts; no raw exports/cache/optimizer."""
    helper().inspect_verified(checkpoint, expected_pins=r["pins"])
    run_id = upload_run_id(r)
    checkpoint_name = Path(checkpoint).parent.name
    if re.fullmatch(r"checkpoint-[0-9]+", checkpoint_name) is None:
        raise ValueError("canonical checkpoint step directory required")
    from huggingface_hub import HfApi
    api = HfApi()
    repo = r["output_repo"]
    if api.repo_info(repo).private is not True:
        raise ValueError("output repository must already be private")
    from huggingface_hub import CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo=f"identity/{run_id}/training/{checkpoint_name}/{name}",
                                     path_or_fileobj=str(Path(checkpoint) / name)) for name in CANONICAL]
    api.create_commit(repo_id=repo, operations=operations, commit_message="Private strict-verified canonical adapter")


def supervise(request_path, mode, checkpoint=None, forward_id=None):
    r = validate_request(read_json(request_path))
    work = Path(r["work"])
    if mode == "train":
        work.mkdir(mode=0o700, parents=True, exist_ok=False)
    elif not work.is_dir():
        raise ValueError("verification requires existing installed runtime")
    os.umask(0o077)
    elapsed = max(0, time.time() - r["overall_started_at"])
    deadline = (None if r.get('operator_managed_stages') else
                time.monotonic() + r["overall_seconds"] - elapsed)
    status = {"status": "starting", "quality_verified": False, "dataset_sha256": DATASET_SHA256}
    started = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true",
               H3_IDENTITY_SUPERVISED="1")
    command = [sys.executable, str(Path(__file__).resolve())]
    try:
        if mode == "train":
            materialize_runtime(r)
            bounded_process(command + ["_train", "--request", str(request_path)], deadline=deadline, env=env)
            training = read_json(work / "training-snapshot.json")
            checkpoint = training["checkpoint"]
            for key in ("training_outcome", "step", "actual_updates", "max_updates", "samples_seen", "epoch"):
                status[key] = training[key]
        status["checkpoint"] = str(checkpoint)
        verify_command = command + ["_verify", "--request", str(request_path), "--checkpoint", str(checkpoint)]
        if forward_id:
            verify_command += ["--forward-id", forward_id]
        bounded_process(verify_command, deadline=deadline, env=env)
        helper().inspect_verified(checkpoint, expected_pins=r["pins"])
        status["status"] = "final_strict_reload_verified_quality_unverified"
        if r.get("output_repo"):
            # Network upload is also under the SAME process-group watchdog, no retry loop.
            bounded_process(command + ["_upload", "--request", str(request_path), "--checkpoint", str(checkpoint)],
                            deadline=deadline, env=dict(os.environ, H3_IDENTITY_SUPERVISED="1"))
            status["uploaded"] = True
    except BaseException as exc:
        status.update(status="stopped_or_failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        raise
    finally:
        status["worker_elapsed_seconds"] = time.monotonic() - started
        status["overall_elapsed_seconds"] = time.time() - r["overall_started_at"]
        report(work / "worker-status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "verify-checkpoint", "_train", "_verify", "_upload"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--forward-id", help="optional frozen sample ID; finite forward is NOT a quality test")
    args = parser.parse_args(argv)
    if args.mode in {"verify-checkpoint", "_verify", "_upload"} and args.checkpoint is None:
        parser.error("--checkpoint required")
    # SIGTERM must unwind the supervisor and kill its entire active stage group.
    def terminate(signum, frame):
        raise KeyboardInterrupt("worker termination requested")
    signal.signal(signal.SIGTERM, terminate)
    if args.mode in {"train", "verify-checkpoint"}:
        supervise(args.request.resolve(), args.mode, args.checkpoint, args.forward_id)
    else:
        if os.environ.get("H3_IDENTITY_SUPERVISED") != "1":
            parser.error("internal stage requires bounded supervisor; use train or verify-checkpoint")
        r = validate_request(read_json(args.request))
        if args.mode == "_train":
            train_stage(r)
        elif args.mode == "_verify":
            verify_stage(r, args.checkpoint, args.forward_id)
        else:
            upload_validated(r, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
