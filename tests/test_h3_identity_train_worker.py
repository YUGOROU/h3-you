"""CPU contracts only; never launch jobs/downloads/install dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from research import hf_identity_train as worker


class WorkerContracts(unittest.TestCase):
    def request(self, root):
        evidence = root / "country.txt"
        evidence.write_text("Reviewed provider statement: assigned GPU executes in JP.")
        frozen = root / "frozen.json"
        frozen.write_text("{}")
        cache = root / "cache"
        cache.mkdir()
        (cache / "index.json").write_text("{}")
        return {
            "schema": "h3-identity-train-request-v1", "max_updates": 100,
            "overall_started_at": time.time(), "overall_seconds": 21600,
            "rate_usd_hour": 5, "spent_and_reserved_usd": 30,
            "source": str(root / "source"), "work": str(root / "work"),
            "model": str(root / "model"), "cache": str(cache),
            "index_sha256": worker.sha256(cache / "index.json"),
            "frozen_selection": str(frozen),
            "pins": {"sources": worker.helper().SOURCE_PINS, "models": {"transformer_ref": {
                "repo_id": "MiniMaxAI/MiniMax-H3", "revision": worker.MODEL_PIN}}},
            "territory_approval": {"approved": True, "approved_by": "reviewer",
                "compute_country": "JP", "evidence_kind": "provider_compute_country_statement",
                "evidence_file": str(evidence), "evidence_sha256": worker.sha256(evidence)},
        }

    def test_request_bounds_and_territory_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.request(Path(d))
            with patch.object(worker, "FROZEN_SHA256", worker.sha256(r["frozen_selection"])):
                worker.validate_request(r)
                for key, value in [("max_updates", 0), ("max_updates", True),
                                   ("overall_seconds", 21601), ("overall_seconds", float("nan")),
                                   ("rate_usd_hour", 11), ("spent_and_reserved_usd", 149)]:
                    bad = dict(r, **{key: value})
                    with self.assertRaises(ValueError):
                        worker.validate_request(bad)
                for country in ("US", "GB", "FR", "KR", "https://hf.co/collections/x", ""):
                    with self.assertRaises(ValueError):
                        worker.territory_gate(dict(r["territory_approval"], compute_country=country))
                with self.assertRaises(ValueError):
                    worker.territory_gate(dict(r["territory_approval"], evidence_kind="collection_url"))

    def test_config_is_actual_schema_subset_not_old_resume(self):
        r = {"model": "/local", "cache": "/cache", "work": "/work", "max_updates": 81}
        c = worker.config_payload(r)
        self.assertEqual(c["optimization"]["steps"], 81)
        self.assertIsNone(c["model"]["load_checkpoint"])
        self.assertFalse(c["training_strategy"]["audio"]["is_generated"])
        self.assertTrue(c["checkpoints"]["identity_peft_export"])
        self.assertEqual(c["checkpoints"]["save_training_state"], "off")
        self.assertEqual(c["data"]["identity_sampling_budget"], 715)
        self.assertEqual(c["data"]["val_split_every"], 0)
        self.assertEqual(c["optimization"]["optimizer_type"], "adamw")

    def test_watchdog_kills_group_on_timeout(self):
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired("stage", 0.1), None, None]
        with patch.object(worker.subprocess, "Popen", return_value=process) as popen, \
             patch.object(worker.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                worker.bounded_process(["fake"], deadline=time.monotonic() + 1)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(kill.call_count, 2)

    def test_expired_budget_never_starts_stage(self):
        with patch.object(worker.subprocess, "Popen") as popen:
            with self.assertRaises(TimeoutError):
                worker.bounded_process(["fake"], deadline=time.monotonic() - 1)
            popen.assert_not_called()

    def test_real_process_timeout_fast_cpu(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            worker.bounded_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                   deadline=time.monotonic() + 0.15)
        self.assertLess(time.monotonic() - start, 5)

    def test_sequential_train_then_verify_no_pilot_and_failure_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r = self.request(root)
            request = root / "request.json"
            request.write_text(json.dumps(r))
            calls = []
            installer = Mock()
            def run(command, **kwargs):
                calls.append(command[2])
                if command[2] == "_train":
                    worker.report(Path(r["work"]) / "training-snapshot.json",
                                  {"checkpoint": "/private/final/identity-peft",
                                   "training_outcome": "bounded_stop", "step": 2,
                                   "actual_updates": 2, "max_updates": 100,
                                   "samples_seen": 2, "epoch": 0})
                else:
                    raise ValueError("strict reload mismatch")
            with patch.object(worker, "validate_request", return_value=r), \
                 patch.object(worker.importlib, "import_module", return_value=installer), \
                 patch.object(worker, "bounded_process", side_effect=run):
                with self.assertRaises(ValueError):
                    worker.supervise(request, "train")
            self.assertEqual(calls, ["_train", "_verify"])
            installer.install.assert_called_once_with(r["source"], Path(r["work"]) / "runtime")
            status = worker.read_json(Path(r["work"]) / "worker-status.json")
            self.assertEqual(status["status"], "stopped_or_failed")
            self.assertFalse(status["quality_verified"])
            self.assertEqual(status["training_outcome"], "bounded_stop")
            self.assertEqual(status["actual_updates"], 2)

    def test_upload_blocked_before_importing_network_client(self):
        fake = Mock()
        fake.inspect_verified.side_effect = ValueError("exported only")
        with patch.object(worker, "helper", return_value=fake):
            with self.assertRaises(ValueError):
                worker.upload_validated({"pins": {}, "output_repo": "private/x"}, "/export")

    def test_uploader_allowlist_and_private_gate(self):
        api = Mock()
        api.repo_info.return_value.private = True
        hub = Mock()
        hub.HfApi.return_value = api
        hub.CommitOperationAdd.side_effect = lambda **kwargs: kwargs
        with patch.object(worker, "helper", return_value=Mock()), \
             patch.dict(sys.modules, {"huggingface_hub": hub}):
            worker.upload_validated({"pins": {}, "output_repo": "private/x", "work": "/private/run1", "run_id": "explicit-run1"},
                                    "/private/checkpoint-10/identity-peft")
            operations = api.create_commit.call_args.kwargs["operations"]
            self.assertEqual({Path(o["path_or_fileobj"]).name for o in operations}, set(worker.CANONICAL))
            self.assertNotIn("adapter.safetensors", str(operations))
            self.assertEqual({o["path_in_repo"] for o in operations}, {
                f"identity/explicit-run1/training/checkpoint-10/{name}" for name in worker.CANONICAL})
            worker.upload_validated({"pins": {}, "output_repo": "private/x",
                                     "work": "/private/run1", "run_id": "explicit-run2"},
                                    "/private/checkpoint-10/identity-peft")
            self.assertTrue(all("identity/explicit-run2/" in o["path_in_repo"]
                                for o in api.create_commit.call_args.kwargs["operations"]))
            api.repo_info.return_value.private = False
            with self.assertRaises(ValueError):
                worker.upload_validated({"pins": {}, "output_repo": "public/x", "work": "/private/run1"},
                                        "/private/checkpoint-10/identity-peft")

    def test_run_namespace_safe_and_full_root_compatibility(self):
        self.assertNotEqual(worker.upload_run_id({"work": "/run-a/training"}),
                            worker.upload_run_id({"work": "/run-b/training"}))
        self.assertEqual(worker.upload_run_id({"work": "/run-a/x/../training"}),
                         worker.upload_run_id({"work": "/run-a/training"}))
        for value in ("", "..", "a/b", "a.b", "é", None, "x" * 129):
            with self.assertRaises(ValueError):
                worker.upload_run_id({"run_id": value})
        for work in ("relative/training", "/"):
            with self.assertRaises(ValueError):
                worker.upload_run_id({"work": work})

    def fake_training(self, root, *, maximum=1000000, remaining=299, mode="normal",
                      update_seconds=1):
        """Exercise train_stage itself with tiny fake pinned modules, no torch import."""
        instances = []
        class Trainer:
            def __init__(self, config):
                instances.append(self)
                self.output_dir = root / "runs" / "training"
                self.state = SimpleNamespace(step=0, samples_seen=0, epoch=0)
                self.optimizer = SimpleNamespace(step=Mock())
                self.original_step = self.optimizer.step
                self.run_logger = SimpleNamespace(log=Mock())
                self.original_log = self.run_logger.log
                self.strategy = SimpleNamespace(compute_loss=Mock())
                self.saved = []
            def setup(self):
                if mode == "setup_failure":
                    raise RuntimeError("setup failed")
            def _core_model(self):
                return SimpleNamespace(parameters=lambda: [SimpleNamespace(
                    requires_grad=True, grad=SimpleNamespace(finite=mode != "bad_gradient"))])
            def save(self, step):
                self.saved.append(step)
                (self.output_dir / f"checkpoint-{step}" / "identity-peft").mkdir(parents=True)
            def train(self):
                if mode == "train_failure":
                    raise RuntimeError("train failed")
                if mode == "zero":
                    return
                for _ in range(maximum):
                    self.optimizer.step()
                    self.state.step += 2 if mode == "mismatch" else 1
                    self.state.samples_seen += 1
                    self.run_logger.log({"loss": float("nan") if mode == "bad_metric" else 1.0},
                                        self.state.step, prefix="train/")
                    if mode == "short_return":
                        return
        class AdapterTensor:
            def __init__(self,value):self.value=value
            def detach(self):return self
            def cpu(self):return self
            def clone(self):return AdapterTensor(self.value)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
            isfinite=lambda t: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: t.finite)),
            equal=lambda a,b:a.value==b.value)
        peft = SimpleNamespace(get_peft_model_state_dict=lambda model,**kwargs:
                               {'adapter':AdapterTensor(instances[-1].state.step)})
        r = {"work": str(root), "pins": {}, "max_updates": maximum,
             "overall_started_at": 1000, "overall_seconds": 2000}
        with patch.dict(sys.modules, {"torch": torch,"peft":peft,
                "h3_trainer.trainer": SimpleNamespace(H3Trainer=Trainer),
                "h3_trainer.checkpointing": SimpleNamespace(
                    checkpoint_dir=lambda output, step: output / f"checkpoint-{step}")}), \
             patch.object(worker, "load_runtime", return_value=Mock()), \
             patch.object(worker, "validate_weights"), patch.object(worker, "verify_dataset"), \
             patch.object(worker, "helper", return_value=Mock()), \
             patch.object(worker, "verify_initial_adapter"), \
             patch.object(worker.time, "time", side_effect=([2000, 2701] if mode == "wall_second"
                                                          else lambda: 3000 - remaining)), \
             patch.object(worker.time, "monotonic", side_effect=iter(
                 i * update_seconds for i in range(20))):
            worker.train_stage(r)
        return worker.read_json(root / "training-snapshot.json"), instances[0]

    def test_actual_logger_wall_stop_exports_current_step_not_huge_cap(self):
        with tempfile.TemporaryDirectory() as d:
            snapshot, trainer = self.fake_training(Path(d))
            self.assertEqual(snapshot["training_outcome"], "bounded_stop")
            self.assertEqual(snapshot["actual_updates"], 1)
            self.assertEqual(snapshot["step"], 1)
            self.assertEqual(snapshot["samples_seen"], 1)
            self.assertIn("checkpoint-1/identity-peft", snapshot["checkpoint"])
            self.assertEqual(trainer.saved, [1])
            trainer.original_step.assert_called_once()
            trainer.original_log.assert_called_once()
            self.assertFalse(snapshot["quality_verified"])

    def test_wall_stop_exports_latest_real_update_after_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            snapshot, trainer = self.fake_training(Path(d), mode="wall_second")
            self.assertEqual(snapshot["actual_updates"], 2)
            self.assertEqual(snapshot["step"], 2)
            self.assertEqual(snapshot["training_outcome"], "bounded_stop")
            self.assertEqual(trainer.saved, [1, 2])
            self.assertIn("checkpoint-2/identity-peft", snapshot["checkpoint"])

    def test_observed_update_time_increases_wall_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            snapshot, _ = self.fake_training(Path(d), remaining=500, update_seconds=260)
            self.assertEqual(snapshot["training_outcome"], "bounded_stop")

    def test_max_updates_completion_honest_even_near_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            snapshot, _ = self.fake_training(Path(d), maximum=3, remaining=1000)
            self.assertEqual(snapshot["training_outcome"], "max_updates_complete")
            self.assertEqual(snapshot["actual_updates"], 3)
        with tempfile.TemporaryDirectory() as d:
            snapshot, _ = self.fake_training(Path(d), maximum=1)
            self.assertEqual(snapshot["training_outcome"], "max_updates_complete")

    def test_no_arbitrary_failure_or_unearned_update_is_accepted(self):
        for mode in ("setup_failure", "train_failure", "zero", "mismatch",
                     "bad_gradient", "bad_metric", "short_return"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.fake_training(Path(d), mode=mode, remaining=1000)
                path = Path(d) / "training-snapshot.json"
                if path.exists():
                    self.assertNotIn("training_outcome", worker.read_json(path))

    def test_shape_and_finite_loss_contract_cpu_fakes(self):
        from types import SimpleNamespace
        class Tensor:
            def __init__(self, shape, finite=True):
                self.shape, self.finite = shape, finite
            def __getitem__(self, key):
                return self
            def numel(self):
                return 1
        torch = SimpleNamespace(isfinite=lambda t: SimpleNamespace(
            all=lambda: SimpleNamespace(item=lambda: t.finite)))
        original = Mock(return_value=SimpleNamespace(total=Tensor(())))
        packed = SimpleNamespace(clean_video=Tensor((1, 2, 4)), clean_audio=Tensor((1, 3, 4)),
                                 video_loss_mask=None, audio_loss_mask=None)
        compute = worker.checked_loss(original, torch)
        compute(Tensor((1, 2, 4)), Tensor((1, 3, 4)), packed)
        for video in (Tensor((2, 4)), Tensor((1, 2, 4), finite=False)):
            with self.assertRaises(ValueError):
                compute(video, Tensor((1, 3, 4)), packed)
        # Real video-only loss: clean audio exists but its generated-row mask is empty.
        packed.audio_loss_mask = SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: False))
        compute(Tensor((1, 2, 4)), None, packed)
        packed.audio_loss_mask = None
        original.return_value.total.finite = False
        with self.assertRaises(ValueError):
            compute(Tensor((1, 2, 4)), Tensor((1, 3, 4)), packed)
        model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=True, grad=None)])
        with self.assertRaises(ValueError):
            worker.check_gradients(model, torch)

    def test_snapshot_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                worker.report(Path(d) / "status.json", {"error": "x" * 40000})

    def test_inventory_digests_and_extra_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text("{}")
            r = {"model": str(root), "model_inventory": {"config.json": worker.sha256(root / "config.json")}}
            worker.validate_weights(r)
            (root / "unexpected").write_text("stale")
            with self.assertRaises(ValueError):
                worker.validate_weights(r)


@unittest.skipUnless(os.environ.get("H3_IDENTITY_TRAIN_OPTIONAL") == "1", "optional existing torch/PEFT env")
class OptionalActualContracts(unittest.TestCase):
    def test_actual_config_and_tiny_peft_roundtrip(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from research import install_identity_checkpoint
        source = os.environ.get("H3_AVIAD_SOURCE", "/tmp/h3-identity-audit/aviad")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            install_identity_checkpoint.install(source, root / "runtime")
            spec = importlib.util.spec_from_file_location("actual_identity_config", root / "runtime/src/h3_trainer/config.py")
            # constants imported from the installed pinned tree, no weights needed.
            sys.path.insert(0, str(root / "runtime/src"))
            try:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                module.H3TrainerConfig.model_validate(worker.config_payload({
                    "model": str(root), "cache": str(root), "work": str(root), "max_updates": 3}))
                class Tiny(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.to_q = torch.nn.Linear(4, 4)
                        self.to_k = torch.nn.Linear(4, 4)
                        self.to_v = torch.nn.Linear(4, 4)
                        self.to_out = torch.nn.Sequential(torch.nn.Linear(4, 4))
                config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                                    target_modules=list(worker.helper().TARGETS))
                model = get_peft_model(Tiny(), config)
                pins = {"sources": worker.helper().SOURCE_PINS, "models": {"transformer_ref": {
                    "repo_id": "MiniMaxAI/MiniMax-H3", "revision": worker.MODEL_PIN}}}
                worker.helper().export_adapter(model, root / "adapter", pins=pins)
                del model
                fresh = get_peft_model(Tiny(), config)
                worker.helper().strict_reload(fresh, root / "adapter", freshly_constructed=True, expected_pins=pins)
                worker.helper().inspect_verified(root / "adapter", expected_pins=pins)
            finally:
                sys.path.pop(0)
                sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
