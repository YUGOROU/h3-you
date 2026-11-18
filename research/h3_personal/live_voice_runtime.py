"""Local resident SMD4 backend; importing/constructing this module does no GPU work.

Public API::

    runtime = LiveVoiceRuntime(model=..., model_manifest=..., identity=...,
        identity_config=..., smd=..., reference=..., archive=..., output=...)
    runtime.start()                   # returns this warmed, resident owner
    result = runtime.render(wav, sha256, turn_directory)
    runtime.close()                   # shuts down VideoGenerator, idempotently

Run start/render/close sequentially in ONE fresh, dedicated GPU-owner process
(caller may use multiprocessing spawn, under its __main__ guard). No RPC, process
launcher, provider allocation or TTS dependency is supplied here. start installs
source overlays and PYTHONPATH before importing FastVideo; its spawned rank
workers inherit source+repository paths and stable manifest/coverage filenames.
Do not start this owner inside a process that already imported FastVideo.

ExternalTTSAdapter wraps an already-loaded native IrodoriResident's synthesize
callback in the separate parent; it neither loads nor owns that resident. Pass
its resulting WAV and SHA to render through the caller's own orchestration.

Requirements: pinned local assets, writable new private output outside the repo,
exactly two H200s, torch2.12.0, fastvideo-kernel0.3.5, quack-kernels0.5.3,
BF16 FA4 dependencies, numpy/Pillow/safetensors and FFmpeg/ffprobe with ALAC.
No downloads, device capture, playback, upload or quality/realtime acceptance.
"""
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import wave

REFERENCE_SHA = '6789db42c765d2611a640bc9b47934b4912584e45c800d2700b1fddf9902faeb'
POLICY = 'resident-visual-cached'


class UnsupportedTemporalContinuation(ValueError):
    """Current H3 Ref2VA has no verified cross-window continuation contract."""


def continuity_contract(window_count, *, require_continuity=False):
    """Describe actual frame/audio boundaries, not an invented continuation API."""
    if type(window_count) is not int or not 1 <= window_count <= 4:
        raise ValueError('one to four windows required')
    if type(require_continuity) is not bool:
        raise ValueError('explicit boolean continuity requirement required')
    if require_continuity and window_count > 1:
        raise UnsupportedTemporalContinuation('independent Ref2VA windows cannot satisfy seamless continuation')
    return dict(mode='independent_ref2va_windows', temporal_state_carried=False,
                previous_tail_conditioning=False, overlap_frames=0,
                motion_continuity_verified=False,
                windows=[dict(index=i, first_frame=i*124, last_frame=(i+1)*124-1,
                              video_start_seconds=i*124/24,
                              audio_start_seconds=i*207/40)
                         for i in range(window_count)])


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _json(path, value, *, immutable=False):
    path = Path(path)
    if immutable:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        path.chmod(0o400)
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value) + '\n')
        temporary.replace(path)


def _command(args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=180).stdout


def _pcm(path):
    return _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                     '-map', '0:a:0', '-c:a', 'pcm_s32le', '-f', 's32le', '-'])


def _metadata(path, frames, *, audio=False):
    from fractions import Fraction
    info = json.loads(_command(['ffprobe', '-v', 'error', '-count_frames',
                               '-show_streams', '-of', 'json', str(path)]))
    videos = [s for s in info['streams'] if s['codec_type'] == 'video']
    sounds = [s for s in info['streams'] if s['codec_type'] == 'audio']
    _require(len(videos) == 1, 'one video stream required')
    v = videos[0]
    _require((int(v['width']), int(v['height']), int(v['nb_read_frames'])) ==
             (252, 448, frames) and Fraction(v['avg_frame_rate']) == 24,
             'portrait frame count/geometry/rate differs')
    _require(len(sounds) == int(audio) and
             (not audio or sounds[0]['codec_name'] == 'alac'), 'audio codec differs')
    _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
              '-f', 'null', '-'])
    return info


class ExternalTTSAdapter:
    """Parent-only adapter; callback is native_resident.synthesize(text, path)."""

    def __init__(self, synthesize):
        if not callable(synthesize):
            raise TypeError('already-loaded native synthesize callback required')
        self._synthesize = synthesize

    def synthesize(self, text, path):
        from research.h3_personal.ref2va_runtime import sha
        path = Path(path).resolve()
        if not isinstance(text, str) or not text.strip() or path.exists():
            raise ValueError('nonempty text and new WAV path required')
        receipt = self._synthesize(text, path)
        from research.run_h3_ref2va import speech_duration
        speech_duration(path)
        return dict(wav=str(path), sha256=sha(path), native_receipt=receipt)


@dataclass(frozen=True)
class RenderResult:
    profile: Path
    review_with_speech: Path


class LiveVoiceRuntime:
    """Independent persistent H3 owner. Construction validates no private assets."""

    def __init__(self, *, model, model_manifest, identity, identity_config, smd,
                 reference, archive, output, continuation=False, comparison_diagnostics=False,
                 decode_parity=False, window_audio_context=False,
                 defer_progress_receipts=False, profile_stages=False,
                 device_audio_validation=False, parallel_decode_waves=False,
                 parallel_decode_acceptance=False, measure_memory=False,
                 trace_policy=None, denoise_policy=None, conditioning_experiment=None,
                 quality_profile=None, conversation_roles=False, visual_intents=False,
                 ordered_references=None):
        if type(conversation_roles) is not bool or (conversation_roles and (quality_profile is None or not continuation)):
            raise ValueError('conversation roles require explicit accepted continuation profile')
        if type(visual_intents) is not bool or (visual_intents and not conversation_roles):
            raise ValueError('visual intents require explicit conversation roles')
        self.ordered_references = None
        if ordered_references is not None:
            if not (conversation_roles and visual_intents and continuation and quality_profile is not None
                    and getattr(quality_profile, 'name', None) == 'reference1024'):
                raise ValueError('ordered references require explicit visual reference1024 continuation/base settings')
            from .ordered_references import verify_local_files, content_sha256
            self.ordered_references = verify_local_files(ordered_references)
            self.ordered_reference_sha256 = content_sha256(self.ordered_references)
            from .candidate_references import build_candidate_binding, candidate_binding_sha256
            self.candidate_reference_binding = build_candidate_binding(self.ordered_references, quality_profile)
            self.candidate_binding_sha256 = candidate_binding_sha256(self.candidate_reference_binding)
        self.conversation_roles = conversation_roles
        self.visual_intents = visual_intents
        self.quality_profile = quality_profile
        if quality_profile is not None:
            from .accepted_profiles import AcceptedQualityProfile
            if type(quality_profile) is not AcceptedQualityProfile:
                raise ValueError('explicit AcceptedQualityProfile required')
            if (continuation is not True or decode_parity or comparison_diagnostics or
                    profile_stages or parallel_decode_acceptance or measure_memory or
                    device_audio_validation or parallel_decode_waves or
                    trace_policy is not None or denoise_policy is not None):
                raise ValueError('accepted quality requires isolated continuation owner')
            manifest = quality_profile.legacy_conditioning_manifest()
            if conditioning_experiment is not None and conditioning_experiment != manifest:
                raise ValueError('accepted conditioning mismatch')
            conditioning_experiment = manifest
        self._conditioning_experiment = None
        if conditioning_experiment is not None:
            from .conditioning_experiment import validate_manifest
            self._conditioning_experiment = validate_manifest(conditioning_experiment)
            if (not continuation or decode_parity or comparison_diagnostics or profile_stages or
                    parallel_decode_acceptance or trace_policy is not None or denoise_policy is not None):
                raise ValueError('conditioning experiment requires isolated eager continuation owner')
        self._trace_policy, self._denoise_policy = None, None
        if trace_policy is not None or denoise_policy is not None:
            from .utilization_policy import policies, validate_mode
            self._trace_policy, self._denoise_policy = policies(trace_policy, denoise_policy)
            validate_mode(self._trace_policy, self._denoise_policy, continuation=continuation,
                          conflicting=bool(decode_parity or comparison_diagnostics or
                                           parallel_decode_acceptance or profile_stages))
        if any(type(v) is not bool for v in (parallel_decode_acceptance, measure_memory)):
            raise ValueError('explicit boolean acceptance/memory opt-in required')
        if parallel_decode_acceptance and (not window_audio_context or decode_parity or
                comparison_diagnostics or device_audio_validation or parallel_decode_waves):
            raise ValueError('parallel acceptance requires isolated v6 baseline owner')
        if measure_memory and not continuation:
            raise ValueError('memory measurement requires continuation owner')
        self._parallel_acceptance_enabled = parallel_decode_acceptance
        self._parallel_acceptance_pending = parallel_decode_acceptance
        self._measure_memory = measure_memory
        if any(type(v) is not bool for v in (window_audio_context, defer_progress_receipts)):
            raise ValueError('explicit boolean candidate opt-in required')
        self._candidate = None
        if window_audio_context or defer_progress_receipts:
            from .continuation_candidate import make_options
            self._candidate = make_options(window_audio_context=window_audio_context,
                                           defer_progress_receipts=defer_progress_receipts)
        if any(type(v) is not bool for v in (device_audio_validation, parallel_decode_waves)):
            raise ValueError('explicit boolean efficiency opt-in required')
        self._efficiency = None
        if device_audio_validation or parallel_decode_waves:
            from .continuation_efficiency import make_options as efficiency_options
            self._efficiency = efficiency_options(device_audio_validation=device_audio_validation,
                                                  parallel_decode_waves=parallel_decode_waves)
            if not window_audio_context:
                raise ValueError('efficiency requires v6 window audio context baseline')
        if self._candidate is not None and (not continuation or decode_parity or comparison_diagnostics):
            raise ValueError('candidate requires isolated continuation opt-in')
        if any(type(v) is not bool for v in (continuation, comparison_diagnostics, decode_parity, profile_stages)):
            raise ValueError('explicit boolean continuation/diagnostic opt-in required')
        if decode_parity and (not continuation or comparison_diagnostics):
            raise ValueError('parity requires isolated continuation opt-in')
        if profile_stages and (not continuation or decode_parity):
            raise ValueError('stage profiling requires non-parity continuation owner')
        self._profile_stages = profile_stages
        self._decode_parity = decode_parity
        if comparison_diagnostics and not continuation:
            raise ValueError('paired diagnostics require continuation overlays')
        self._comparison_diagnostics = comparison_diagnostics
        self._continuation_enabled = continuation
        self.paths = {k: Path(v).resolve() for k, v in dict(
            model=model, model_manifest=model_manifest, identity=identity,
            identity_config=identity_config, smd=smd, reference=reference,
            runtime_archive=archive, output=output).items()}
        self.generator = None
        self._state = 'new'
        self._lock = threading.Lock()
        self._requests = 0

    def start(self):
        """Explicit GPU initialization + one discarded 5.175s zero-PCM clip; return self.

        Cold packed instrumentation may finish only on the first real render;
        no warm-only reports are assumed to exist after this single request.
        """
        with self._lock:
            _require(self._state == 'new', 'start is single-use')
            self._state = 'starting'
            try:
                self._start()
                self._state = 'ready'
                return self
            except BaseException:
                self._shutdown()
                raise

    def _start(self):
        from types import SimpleNamespace
        from research.run_h3_ref2va import verify_inputs, FASTVIDEO_REV
        from research.prepare_h3_realtime_first import validate_archive
        from research.h3_personal.fast_ref2va import install, build_adapter, config_for_layout, generator_config
        from research.h3_personal.external_audio import install as install_audio
        from research.h3_personal.ref2va_attention import configure_environment
        root = Path(__file__).resolve().parents[2]
        out = self.paths['output']
        _require(not out.is_relative_to(root), 'private output must be outside repository')
        _require(not any(k == 'fastvideo' or k.startswith('fastvideo.') for k in sys.modules),
                 'start requires a fresh GPU owner without FastVideo imports')
        args = SimpleNamespace(**self.paths, reference_sha256=REFERENCE_SHA,
                               fast_ref_smd=True, face_crop=None)
        self.proof = verify_inputs(args)
        validate_archive(self.paths['runtime_archive'])
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        with tarfile.open(self.paths['runtime_archive']) as archive:
            archive.extractall(out / 'runtime', filter='data')
        source = out / 'runtime' / ('FastVideo-' + FASTVIDEO_REV)
        overlay = install(source)
        overlay['locked_audio'] = install_audio(source)
        if self._continuation_enabled:
            from research.h3_personal.fast_continuation_overlay import install as install_continuation
            overlay['continuation'] = install_continuation(source)
        _json(out / 'loader-overlay.json', overlay, immutable=True)
        self.coverage = out / 'coverage'
        self.coverage.mkdir()
        self.window_file = self.coverage / 'current-locked-window.json'
        from .conditioning_experiment import EXPERIMENT_ENV, ACCEPTED_PROFILE_ENV
        os.environ.pop(EXPERIMENT_ENV, None)
        os.environ.pop(ACCEPTED_PROFILE_ENV, None)
        from .conversation_conditioning import (ORDERED_CONTEXT_ENV, ordered_context_record, role_prompt)
        os.environ.pop(ORDERED_CONTEXT_ENV, None)
        if getattr(self, 'ordered_references', None) is not None:
            from .ordered_references import verify_local_files
            from .conditioning_experiment import _publish_private
            from dataclasses import replace
            import hashlib
            selected = verify_local_files(self.ordered_references,
                expected_content_sha256=self.ordered_reference_sha256)
            self.ordered_context_file = self.coverage / 'ordered-reference-context.json'
            _publish_private(self.ordered_context_file, ordered_context_record(selected), immutable=True)
            from .candidate_references import build_candidate_binding
            if build_candidate_binding(selected, self.quality_profile) != self.candidate_reference_binding:
                raise ValueError('candidate BASE settings/content binding drift')
            self.candidate_binding_file = self.coverage / 'candidate-reference-binding.json'
            _publish_private(self.candidate_binding_file, self.candidate_reference_binding, immutable=True)
            os.environ[ORDERED_CONTEXT_ENV] = str(self.ordered_context_file)
            # Accepted profile remains bound to ORIGINAL baseline reference; the
            # separate context marks all new content visually UNREVIEWED.
            self.quality_profile = replace(self.quality_profile,
                prompt_sha256=hashlib.sha256(role_prompt('speech').encode()).hexdigest())
            self.proof.update(ordered_reference_sha256=self.ordered_reference_sha256,
                              candidate_binding_sha256=self.candidate_binding_sha256,
                              candidate_reference_binding=self.candidate_reference_binding,
                              candidate_visual_acceptance='unreviewed')
        if self.quality_profile is not None:
            self.quality_profile_file = self.coverage / 'accepted-quality-profile.json'
            from .conditioning_experiment import _publish_private
            _publish_private(self.quality_profile_file, self.quality_profile.wire_record())
            os.environ[ACCEPTED_PROFILE_ENV] = str(self.quality_profile_file)
        if self._conditioning_experiment is not None:
            self.conditioning_file = self.coverage / 'conditioning-experiment.json'
            from .conditioning_experiment import _publish_private
            _publish_private(self.conditioning_file, self._conditioning_experiment)
            os.environ[EXPERIMENT_ENV] = str(self.conditioning_file)
        if self._continuation_enabled:
            self.continuation_file = self.coverage / 'current-continuation-window.json'
            _json(self.continuation_file, dict(mode='independent'))
            os.environ['H3_SMD_CONTINUATION_WINDOW_FILE'] = str(self.continuation_file)
        else:
            os.environ.pop('H3_SMD_CONTINUATION_WINDOW_FILE', None)
        env = dict(PYTHONPATH=str(source) + os.pathsep + str(root),
                   H3_REF2VA_COVERAGE_DIR=str(self.coverage), H3_SMD_REF2VA='1',
                   H3_SMD_LOCKED_AUDIO='1', H3_SMD_LOCKED_AUDIO_WINDOW_FILE=str(self.window_file),
                   FASTVIDEO_H3_REF2VA_MEMO_ENTRIES='1', H3_REF2VA_RUNTIME_POLICY=POLICY,
                   H3_REF2VA_REGIONAL_COMPILE='0', TORCHDYNAMO_SUPPRESS_ERRORS='0',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        configure_environment('fa4-packed', env)
        os.environ.update(env)
        os.environ.pop('H3_REF2VA_STAGE_PROFILE_DIR', None)
        os.environ.pop('H3_REF2VA_COMPARISON_PROFILE', None)
        os.environ.pop('H3_SMD_DECODE_PARITY', None)
        os.environ.pop('H3_SMD_MEASURE_MEMORY', None)
        if self._measure_memory:
            os.environ['H3_SMD_MEASURE_MEMORY'] = '1'
        os.environ.pop('H3_SMD_PARALLEL_DECODE_WAVES', None)
        if self._efficiency is not None and self._efficiency['decode_schedule'] == 'two-clip-waves-v1':
            os.environ['H3_SMD_PARALLEL_DECODE_WAVES'] = '1'
        if self._decode_parity:
            os.environ['H3_SMD_DECODE_PARITY'] = '1'
        if self._comparison_diagnostics or self._profile_stages:
            os.environ['H3_REF2VA_STAGE_PROFILE_DIR'] = str(out / 'stage-profile')
        if self._comparison_diagnostics:
            os.environ['H3_REF2VA_COMPARISON_PROFILE'] = '1'
        sys.path[:0] = [str(source), str(root)]
        # All overlays and child-worker environment are now installed.
        import fastvideo
        import fastvideo.api as api
        import torch
        from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference
        _require(Path(fastvideo.__file__).resolve().parent == source / 'fastvideo', 'wrong source imported')
        for package, version in {'torch': '2.12.0', 'fastvideo-kernel': '0.3.5',
                                 'quack-kernels': '0.5.3'}.items():
            _require(importlib.metadata.version(package).split('+')[0] == version,
                     'wrong pinned dependency: ' + package)
        _require(torch.cuda.device_count() == 2 and all(
            'H200' in torch.cuda.get_device_name(i) for i in range(2)), 'exactly two H200s required')
        torch.set_num_threads(4)
        self.api, self.reference_type, self.torch = api, MiniMaxH3Reference, torch
        self.config = config_for_layout('portrait')
        if getattr(self, 'conversation_roles', False):
            from .conversation_conditioning import (ROLES_ENV, CURRENT_ROLE_ENV, roles_manifest,
                                                    role_record, SPEECH_PROMPT)
            from .conditioning_experiment import _publish_private
            _require(self.config.prompt == SPEECH_PROMPT, 'initial speech prompt required')
            if getattr(self, 'ordered_references', None) is not None:
                from dataclasses import replace
                from .conversation_conditioning import role_prompt
                self.config = replace(self.config, prompt=role_prompt('speech'))
            self.roles_file = self.coverage / 'conversation-roles.json'
            self.current_role_file = self.coverage / 'conversation-current-role.json'
            _publish_private(self.roles_file, roles_manifest(visual_intents=self.visual_intents), immutable=True)
            _publish_private(self.current_role_file, role_record('speech', visual_intents=self.visual_intents))
            os.environ[ROLES_ENV] = str(self.roles_file)
            os.environ[CURRENT_ROLE_ENV] = str(self.current_role_file)
        adapter = out / 'derived-private' / 'adapter_model.safetensors'
        self.derived = build_adapter(self.paths['smd'], self.paths['identity'],
                                     self.paths['identity_config'], adapter,
                                     canonical_fingerprint=self._comparison_diagnostics)
        from research.run_h3_realtime_first import encoder_preflight
        encoder_preflight(out / 'encoder-probe')
        warm = out / 'warmup.wav'
        with wave.open(str(warm), 'wb') as handle:
            handle.setparams((2, 2, 32000, 0, 'NONE', 'not compressed'))
            handle.writeframes(bytes(165600 * 2 * 2))  # exactly 5.175s
        from research.h3_personal.ref2va_runtime import sha
        plan = self._plan(warm, sha(warm))
        self._bind(plan, 0)
        self.generator = fastvideo.VideoGenerator.from_config(generator_config(
            api, self.paths['model'], adapter, regional_compile=False, runtime_policy=POLICY))
        self._generate()  # discard frames; completes deferred load, merge, condition cache
        self._loaded()
        self._locked(plan, 0)

    def _plan(self, wav, digest):
        from research.run_h3_ref2va import speech_duration
        from research.h3_personal.external_audio import plan_windows
        from research.h3_personal.ref2va_runtime import sha
        _require(sha(wav) == digest, 'supplied WAV SHA differs')
        with wave.open(str(wav), 'rb') as handle:
            _require(handle.getcomptype() == 'NONE' and handle.getsampwidth() in (2, 3, 4),
                     'lossless integer PCM WAV required')
        return plan_windows(speech_duration(wav), wav=wav, sha256_value=digest, max_windows=4)

    def _bind(self, plan, index):
        from research.h3_personal.external_audio import window_manifest
        for rank in (0, 1):
            (self.coverage / f'locked-audio-rank-{rank}-window-{index}.json').unlink(missing_ok=True)
        _json(self.window_file, window_manifest(plan, index))

    def _generate(self, *, continuation=False):
        from research.h3_personal.ref2va_runtime import request
        quarantine=getattr(self,'_h3_live_native_quarantine',None)
        if quarantine is not None:
            command=json.loads(Path(self.continuation_file).read_text())
            control=command.get('conversation_control',{})
            _require(continuation and set(command)=={'conversation_control'}
                     and control.get('conversation_id')==quarantine._id
                     and control.get('action') in ('abandon','retire'),
                     'uncertain live owner permits only explicit health-checked release')
        if continuation and not self._continuation_enabled:
            raise ValueError('continuation must be enabled before worker spawn')
        self._validate_quality_binding()
        selected = getattr(self, 'ordered_references', None)
        if selected is None:
            req = request(self.api, self.reference_type, self.paths['reference'], self.config)
        else:
            self._validate_ordered_reference_binding()
            req = request(self.api, self.reference_type, selected['references'][0]['path'],
                          self.config, ordered_references=selected)
        if continuation:
            _require(req.sampling.seed == 314159, 'continuation fixed seed differs')
        else:
            req.sampling.seed += max(0, self._requests - 1)  # warmup does not consume real seed314159
        self._last_seed = req.sampling.seed
        begin = time.monotonic()
        result = self.generator.generate(req)
        self.torch.cuda.synchronize()
        self._requests += 1
        return result, (begin, time.monotonic())

    def _read(self, stem):
        return [json.loads((self.coverage / f'{stem}-rank-{rank}.json').read_text()) for rank in (0, 1)]

    def _loaded(self):
        loaded = self._read('coverage')
        for rank, receipt in enumerate(loaded):
            _require(receipt == dict(identity_modules=208, total_modules=312, unmatched=0,
                rank='mixed128-144', partition='transformer_ref', worker_rank=rank),
                'both-rank original64 identity coverage required')
        merges = self._read('merge')
        _require(all(r == dict(active=312, merged=312, dynamic=0) for r in merges), 'static merge differs')
        attention = self._read('attention')
        _require(all(r.get('backend') == '4' and r.get('cuda_kernel_events') and
                     r.get('dtype') == 'bfloat16' for r in attention), 'BF16 FA4 smoke missing')
        return dict(loaded_coverage=loaded, static_merge_receipts=merges, attention_receipts=attention)

    def _locked(self, plan, index):
        from research.h3_personal.external_audio import verify_receipts, window_manifest
        # Verify just this generation; earlier windows may belong to another turn.
        # Helper requires contiguous indices, so inspect only the final pair here.
        receipts = verify_receipts(self.coverage, index + 1)[-2:]
        manifest = window_manifest(plan, index)
        for receipt in receipts:
            _require(all(receipt.get(key) == value for key, value in dict(
                wav=manifest['wav'], wav_sha256=manifest['sha256'],
                latent_start=manifest['latent_start'], latent_count=manifest['latent_count'],
                expected_rows=manifest['expected_rows'], encoded_rows=manifest['expected_rows'],
                audio_velocity_discarded=True, generated_audio_decode_skipped=True).items()),
                'speech binding/rows differ')
        return receipts

    def render(self, wav, sha, turnDir, *, require_continuity=False):
        """Render full supplied speech (<=4 windows). Return RenderResult paths.

        turnDir must be new; failures leave diagnostic artifacts, never overwrite
        a turn. No head trim, shortest mux, speech shortening or time stretching.
        require_continuity=True rejects unsupported multi-window speech before
        GPU work; legacy independent-window mode remains diagnostic, not seamless.
        This guard does not implement temporal continuation.
        """
        with self._lock:
            _require(self._state == 'ready', 'runtime is not ready')
            if type(require_continuity) is not bool:
                raise ValueError('explicit boolean continuity requirement required')
            if require_continuity:
                # Reject unsupported work BEFORE touching the GPU. An input
                # admission rejection must not shut down the warmed model.
                plan = self._plan(Path(wav).resolve(), sha)
                continuity_contract(plan['windows_required'], require_continuity=True)
            try:
                return self._render(Path(wav).resolve(), sha, Path(turnDir).resolve())
            except BaseException:
                # Partial GPU failures must not silently reuse an uncertain resident.
                self._shutdown()
                raise

    def _validate_ordered_reference_binding(self):
        selected = getattr(self, 'ordered_references', None)
        if selected is None:
            return
        from .ordered_references import verify_local_files
        from .conversation_conditioning import read_ordered_context
        selected = verify_local_files(selected, expected_content_sha256=self.ordered_reference_sha256)
        context = read_ordered_context()
        from .candidate_references import build_candidate_binding, validate_candidate_binding
        _require(build_candidate_binding(selected, self.quality_profile) == self.candidate_reference_binding
                 and validate_candidate_binding(json.loads(self.candidate_binding_file.read_text()))
                     == self.candidate_reference_binding,
                 'candidate BASE settings/content artifact drift')
        _require(context is not None and context['ordered_references'] == selected
                 and context['identity_sha256'] == self.ordered_reference_sha256,
                 'ordered-reference runtime context drift')

    def _validate_quality_binding(self, profile=None):
        profile = getattr(self, 'quality_profile', None) if profile is None else profile
        if profile is None:
            return
        import hashlib
        from .ref2va_runtime import sha
        profile.validate_binding(ordered_source_sha256=(sha(self.paths['reference']),),
            prompt_sha256=hashlib.sha256(self.config.prompt.encode()).hexdigest(),
            processor_sha256=profile.processor_sha256,
            geometry=(self.config.width, self.config.height))
        _require(self.config.num_frames == profile.window and self.config.fps == profile.fps,
                 'accepted window/fps mismatch')

    def configure_quality_profile(self, profile):
        """Switch only the accepted edge, without regeneration or worker restart."""
        from .accepted_profiles import AcceptedQualityProfile
        from .conditioning_experiment import _publish_private
        with self._lock:
            _require(getattr(self, 'ordered_references', None) is None,
                     'candidate active shortedge is immutable; no implicit512 route')
            _require(self._state == 'ready' and self.quality_profile is not None,
                     'ready accepted owner required')
            _require(type(profile) is AcceptedQualityProfile, 'accepted profile required')
            _require(json.loads(Path(self.continuation_file).read_text()).get('mode') ==
                     'independent', 'no active continuation owner required')
            old = self.quality_profile
            from dataclasses import replace
            _require(replace(profile, name=old.name) == old, 'warm binding changes prohibited')
            self._validate_quality_binding(profile)
            policy = profile.legacy_conditioning_manifest()
            try:
                _publish_private(self.conditioning_file, policy)
                _publish_private(self.quality_profile_file, profile.wire_record())
            except BaseException:
                try:
                    _publish_private(self.conditioning_file, old.legacy_conditioning_manifest())
                    _publish_private(self.quality_profile_file, old.wire_record())
                except BaseException:
                    self._state = 'degraded'
                raise
            self.quality_profile, self._conditioning_experiment = profile, policy

    def configure_conversation_role(self, role):
        """Select a prewarmed public role between completed source Sessions."""
        with self._lock:
            self._switch_role(role)

    def _switch_role(self, role, *, visual_binding=None):
        """Caller holds the owner lock; never reloads Qwen or the DiT."""
        from dataclasses import replace
        from .conversation_conditioning import role_prompt, role_record, visual_prompt, validate_visual_record
        from .conditioning_experiment import _publish_private
        _require(getattr(self, 'conversation_roles', False) and self._state == 'ready',
                 'ready conversation owner required')
        _require(json.loads(Path(self.continuation_file).read_text()).get('mode') ==
                 'independent', 'completed source Session required')
        old_profile, old_config = self.quality_profile, self.config
        old_role = json.loads(Path(self.current_role_file).read_text())
        if getattr(self, 'visual_intents', False):
            record = validate_visual_record(visual_binding or role_record(role, visual_intents=True,
                active=old_role.get('visual_intent'), intent_epoch=old_role.get('intent_epoch', 0),
                target_epoch=old_role.get('target_epoch', 0)))
            if record['role'] != role or record['target_epoch'] < old_role.get('target_epoch', 0):
                raise ValueError('stale/foreign visual role binding')
            prompt = visual_prompt(role, record['visual_intent'])
        else:
            if visual_binding is not None:raise ValueError('visual role not enabled')
            record = role_record(role)
            prompt = role_prompt(role)
        config = replace(old_config, prompt=prompt)
        profile = replace(old_profile, prompt_sha256=record['prompt_sha256'])
        self.config = config
        try:
            _publish_private(self.current_role_file, record)
            self._validate_quality_binding(profile)
            _publish_private(self.quality_profile_file, profile.wire_record())
        except BaseException:
            self.config = old_config
            try:
                _publish_private(self.quality_profile_file, old_profile.wire_record())
                _publish_private(self.current_role_file, old_role)
            except BaseException:
                self._state = 'degraded'
            raise
        self.quality_profile = profile

    def configure_conditioning(self, short_edge):
        """Switch a precomputed experimental condition at a completed-turn boundary."""
        from .conditioning_experiment import validate_manifest
        with self._lock:
            _require(self._state == 'ready' and self._conditioning_experiment is not None,
                     'ready experimental owner required')
            _require(json.loads(Path(self.continuation_file).read_text()).get('mode') ==
                     'independent', 'no active continuation owner required')
            _require(self.quality_profile is None, 'use configure_quality_profile for accepted owner')
            policy = validate_manifest(dict(self._conditioning_experiment, active_short_edge=short_edge))
            from .conditioning_experiment import _publish_private
            _publish_private(self.conditioning_file, policy)
            self._conditioning_experiment = policy

    def configure_utilization(self, trace_policy=None, denoise_policy=None):
        """Arm the next fresh turn without replacing the resident generator."""
        with self._lock:
            _require(self._state == 'ready', 'ready/no active owner required')
            _require(not getattr(self, '_conditioning_experiment', None) or
                     (trace_policy is None and denoise_policy is None),
                     'conditioning experiment cannot arm tracing/local compile')
            trace, denoise = None, None
            if trace_policy is not None or denoise_policy is not None:
                from .utilization_policy import policies, validate_mode
                trace, denoise = policies(trace_policy, denoise_policy)
                validate_mode(trace, denoise, continuation=self._continuation_enabled,
                              conflicting=bool(self._decode_parity or self._comparison_diagnostics or
                                               self._parallel_acceptance_enabled or self._profile_stages))
            _require(not getattr(self, '_denoise_policy', None) or denoise is not None,
                     'installed local regions cannot be disabled on a warm model')
            if hasattr(self, 'continuation_file'):
                _require(json.loads(Path(self.continuation_file).read_text()).get('mode') ==
                         'independent', 'no active continuation owner required')
            self._trace_policy, self._denoise_policy = trace, denoise

    def configure_efficiency(self, *, device_audio_validation, parallel_decode_waves,
                             defer_progress_receipts):
        """Diagnostic arm boundary only; no reload/reset or worker env mutation.

        Caller must validate both-rank parity receipts before this explicit arm
        switch. Mode restoration after failure is not acceptance or recovery.
        """
        from .continuation_candidate import make_options
        from .continuation_efficiency import make_options as efficiency_options
        with self._lock:
            _require(self._state == 'ready' and self._parallel_acceptance_enabled and
                     not self._parallel_acceptance_pending,
                     'completed acceptance owner required before arm switch')
            candidate = make_options(window_audio_context=True,
                                     defer_progress_receipts=defer_progress_receipts)
            efficiency = efficiency_options(device_audio_validation=device_audio_validation,
                                             parallel_decode_waves=parallel_decode_waves)
            self._candidate, self._efficiency = candidate, efficiency

    def render_continuation(self, wav, sha, turnDir, *, on_segment=None):
        """Explicit opt-in AV latent-tail path; no live transport or quality claim.

        on_segment sees each verified video-only suffix file and rational PTS.
        Source speech is muxed once at the end. A failure holds this owner in
        degraded state for diagnosis, never automatically shuts down/reloads it.
        """
        from research.h3_personal.fast_continuation_render import plan, render
        with self._lock:
            _require(self._state == 'ready', 'runtime is not ready')
            if not self._continuation_enabled:
                raise ValueError('continuation must be opted into before start')
            if on_segment is not None and not callable(on_segment):
                raise ValueError('callable segment observer required')
            target = Path(turnDir).resolve()
            root = Path(__file__).resolve().parents[2]
            if target.is_relative_to(root) or target.exists():
                raise ValueError('new private output outside repository required')
            # Pure admission errors must not discard the warmed runtime.
            plan(Path(wav).resolve(), sha)
            try:
                result = render(self, Path(wav).resolve(), sha, target, on_segment=on_segment)
                self._parallel_acceptance_pending = False
                return result
            except BaseException:
                self._state = 'degraded'
                raise

    def render_incremental_av(self, wav, sha, turnDir, output, *, timeout=60, paced=False):
        """Opt-in committed AV into one headless clock; never device playback.

        Final review mux remains an audit artifact, not a second audio producer.
        This method does not add cross-source listening/visual-tail handoff.
        """
        from .incremental_av import render_incremental_av
        return render_incremental_av(self, wav, sha, turnDir, output,
                                     timeout=timeout, paced=paced)

    def render_conversation_av(self, listening_wavs, speech_wav, turnDir, output, *,
                               timeout=300, paced=True):
        """One finite listening -> speech -> listening scenario into one AV sink.

        Explicit opt-in only: accepted quality profile, conversation roles and a
        continuation owner. Never device playback, Discord or a new conversation
        E2E. A local sink/encode/preparation failure abandons the in-progress
        session but never reloads or restarts the resident model.
        """
        from .conversation_av import ConversationAVStream
        from .conversation_scenario import (
            ConversationRenderer, plan as scenario_plan, read_window_receipts)
        from .conversation_sequence import ConversationSequence
        from .accepted_profiles import AcceptedQualityProfile
        with self._lock:
            _require(self._state in ('ready', 'degraded'), 'runtime is not started')
            _require(self._continuation_enabled and getattr(self, 'conversation_roles', False),
                     'explicit conversation owner required')
            _require(type(self.quality_profile) is AcceptedQualityProfile,
                     'accepted quality profile required')
            _require(json.loads(Path(self.continuation_file).read_text()).get('mode') ==
                     'independent', 'completed source Session required')
            import uuid
            scenario = scenario_plan(conversation_id='conversation-' + uuid.uuid4().hex,
                                     profile_sha256=self.quality_profile.digest,
                                     listening_wavs=listening_wavs, speech_wav=speech_wav)
            binding = scenario.binding()
            root = Path(self.paths['output']).resolve(strict=True)
            repo = Path(__file__).resolve().parents[2]
            turn = Path(turnDir).resolve()
            dest = Path(output).absolute()
            parent = dest.parent.resolve(strict=True)
            dest = parent / dest.name
            _require(not root.is_relative_to(repo) and turn.is_relative_to(root) and
                     turn != root and not turn.exists() and dest.is_relative_to(root) and
                     not dest.is_relative_to(turn) and not dest.exists() and
                     not dest.is_symlink(), 'fresh private turn and separate output required')
            _require(not root.stat().st_mode & 0o077 and not parent.stat().st_mode & 0o077,
                     'private output directories required')
            turn.mkdir(mode=0o700)
            stream = ConversationAVStream(root, binding,
                width=self.quality_profile.exported_geometry[0],
                height=self.quality_profile.exported_geometry[1], timeout=timeout, paced=paced)
            speech = binding.sources[0]
            sequence = ConversationSequence(speech_digest=speech.sha256,
                                            sample_rate=speech.sample_rate, samples=speech.samples,
                                            channels=speech.channels)
            backend = _ConversationBackend(self, scenario)
            renderer = ConversationRenderer(scenario, binding, stream, turn, backend,
                width=self.quality_profile.exported_geometry[0],
                height=self.quality_profile.exported_geometry[1],
                manifest_path=self.continuation_file,
                receipts=lambda epoch, index: read_window_receipts(self.coverage, scenario, epoch, index),
                sequence=sequence, before_window=backend.before_window)
            published = False
            try:
                receipt = renderer.run()
                completed = Path(receipt['output'])
                _require(completed.is_file(), 'conversation sink output missing')
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                published = True
                with os.fdopen(fd, 'wb') as dst, completed.open('rb') as src:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
                dest.chmod(0o400)
                receipt['output'] = str(dest)
                receipt['output_sha256'] = _sha_bytes(dest)
                receipt['generation'] = backend.record
                return receipt
            except BaseException as error:
                if published:
                    dest.unlink(missing_ok=True)
                backend.abandon(self, error)
                raise
            finally:
                stream.cancel()
                if json.loads(Path(self.continuation_file).read_text()).get('mode') != 'independent':
                    try:
                        _json(self.continuation_file, dict(mode='independent'))
                    except BaseException:
                        self._state = 'degraded'

    def _render(self, wav, digest, turn):
        import numpy as np
        from research.h3_personal.ref2va_runtime import sha
        from research.h3_personal.fast_ref2va import segment_timing
        from research.run_h3_realtime_first import encode_rgb_segment
        root = Path(__file__).resolve().parents[2]
        _require(not turn.is_relative_to(root), 'turn output must be outside repository')
        plan = self._plan(wav, digest)
        count = plan['windows_required']
        _require(count * 124 / 24 >= plan['speech_seconds'], 'video cannot cover full speech without shortening')
        turn.mkdir(mode=0o700, parents=True, exist_ok=False)
        # Snapshot the input too: never depend on a mutable parent TTS output.
        speech = turn / 'speech.wav'
        with speech.open('xb') as target:
            target.write(wav.read_bytes())
        speech.chmod(0o400)
        _require(sha(speech) == digest, 'speech changed while staging')
        plan = self._plan(speech, digest)
        _json(turn / 'speech-plan.json', plan, immutable=True)
        segments, windows, intervals = [], [], []
        for index in range(count):
            self._bind(plan, index)
            result, interval = self._generate()
            receipts = self._locked(plan, index)
            loaded = self._loaded()
            cache = self._read('cache')
            _require(all(r == {'hits': self._requests - 1} for r in cache), 'condition-cache hit differs')
            packed = self._read('packed')  # first real warm_forward finishes cold probe
            _require(all(r.get('warm_counter_removed') and r.get('packed_route_executed')
                         for r in packed), 'packed FA4 cold route not proved/removed')
            residency = self._read('residency')
            _require(all(r.get('policy') == POLICY and r.get('audio_decode_skipped')
                         for r in residency), 'resident visual policy differs')
            snapshot = dict(window_index=index, seed=self._last_seed,
                            fixed_reference_sha256=self.proof['reference_sha256'],
                            wav_sha256=digest, locked_audio_receipts=receipts,
                            **loaded, conditioning_cache_receipts=cache,
                            packed_route_receipts=packed, residency_receipts=residency)
            _json(turn / f'window-{index:03d}-receipts.json', snapshot, immutable=True)
            frames = np.stack([np.asarray(f).copy() for f in result.frames])
            _require(frames.dtype == np.uint8 and frames.shape == (124, 448, 256, 3),
                     'native portrait output shape differs')
            frames = frames[:, :, 2:-2, :].copy()
            _require(frames.shape == (124, 448, 252, 3), 'export shape differs')
            segment = turn / f'segment-{index:03d}.mp4'
            encode_rgb_segment(frames.tobytes(), segment, segment_timing(index), width=252, height=448)
            _metadata(segment, 124)
            segments.append(segment)
            windows.append(snapshot)
            intervals.append(interval)
        listing = turn / 'concat.txt'
        listing.write_text(''.join(f"file '{p.name}'\nduration {124 / 24:.12f}\n" for p in segments))
        joined = turn / 'joined.mp4'
        _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-f', 'concat',
                  '-safe', '1', '-i', str(listing), '-map', '0:v:0', '-c:v', 'copy', '-an', str(joined)])
        _metadata(joined, 124 * count)
        review = turn / 'review-with-speech.mp4'
        _command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-n', '-i', str(joined),
                  '-i', str(speech), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                  '-c:a', 'alac', str(review)])
        _metadata(review, 124 * count, audio=True)
        _require(sha(wav) == digest and sha(speech) == digest and _pcm(speech) == _pcm(review),
                 'speech SHA or lossless mux PCM differs')
        profile = dict(schema='h3-resident-live-voice-v1', input_proof=self.proof,
            derived_adapter_receipt=self.derived, locked_audio_plan=plan, windows=windows,
            generation_intervals=intervals, generation_seconds=[b-a for a,b in intervals],
            warmup_discarded=True, warm_measurement_available=False,
            latency_scope='resident server generation only; no playback or realtime acceptance',
            expected_forwards_per_segment=4, unique_frames=124*count, runtime_policy=POLICY,
            attention_mode='fa4-packed', layout='portrait', visual_only=True,
            generated_audio_discarded=True, speech_pcm_exact=True, full_decode_verified=True,
            shared_head_trim=False, independent_ref2va_clips=True,
            temporal_continuity=continuity_contract(count), identity_quality_verified=False,
            lip_sync_verified=False, realtime_complete=False, uploaded=False,
            files={p.name: sha(p) for p in [speech, *segments, joined, review]})
        path = turn / 'profile.json'
        _json(path, profile, immutable=True)
        for p in [*segments, joined, review]:
            p.chmod(0o400)
        return RenderResult(profile=path, review_with_speech=review)

    def _shutdown(self):
        generator, self.generator = self.generator, None
        self._state = 'closed'
        try:
            if generator is not None:
                generator.shutdown()
        finally:
            from .conditioning_experiment import EXPERIMENT_ENV, ACCEPTED_PROFILE_ENV
            from .conversation_conditioning import ROLES_ENV, CURRENT_ROLE_ENV, ORDERED_CONTEXT_ENV
            for variable, attribute in ((ORDERED_CONTEXT_ENV, 'ordered_context_file'),
                                        (ROLES_ENV, 'roles_file'),
                                        (CURRENT_ROLE_ENV, 'current_role_file'),
                                        (EXPERIMENT_ENV, 'conditioning_file'),
                                        (ACCEPTED_PROFILE_ENV, 'quality_profile_file')):
                owned = getattr(self, attribute, None)
                if owned is not None and os.environ.get(variable) == str(owned):
                    os.environ.pop(variable, None)

    def close(self):
        """Release the resident VideoGenerator; never owns/closes external TTS."""
        with self._lock:
            self._shutdown()


def _sha_bytes(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _ConversationBackend:
    """Real per-window native generation for one finite conversation scenario.

    Owns no model lifecycle: it never reloads, restarts or silently changes
    profile/precision. Only the exact exported frames of this window are returned.
    """
    def __init__(self, runtime, scenario, stack=None):
        self.runtime = runtime
        self.scenario = scenario
        self.record = []
        self.stack = stack

    def before_window(self, epoch, index):
        runtime = self.runtime
        if index == 0:
            if epoch > 0:
                # The previous source is complete: restore independent mode.
                _json(runtime.continuation_file, dict(mode='independent'))
            # Every source binds its own precomputed role before publication;
            # the first source is listening, not the initial speech role.
            runtime._switch_role(self.scenario.sources[epoch].role)

    def _stack(self, frames):
        """Stack per-frame pixels into (count,448,256,3); numpy only in production."""
        if self.stack is not None:
            return self.stack(frames)
        import numpy as np
        return np.stack([np.asarray(f).copy() for f in frames])

    def render_window(self, epoch, index, final):
        runtime = self.runtime
        begin = time.monotonic()
        result, interval = runtime._generate(continuation=True)
        frames = self._stack(result.frames)
        if frames.ndim != 4 or frames.shape[1:] != (448, 256, 3):
            raise ValueError(f'conversation native output shape differs: {frames.shape}')
        exported = frames[:, :, 2:-2, :].copy(order='C')
        if exported.shape[1:] != (448, 252, 3) or not exported.flags['C_CONTIGUOUS']:
            raise ValueError('conversation export geometry differs')
        self.record.append(dict(source_epoch=epoch, window_index=index,
                                emitted_frames=int(exported.shape[0]),
                                generation_interval=list(interval),
                                generate_seconds=interval[1] - interval[0],
                                wall_seconds=time.monotonic() - begin,
                                request_index=runtime._requests))
        return dict(raw=exported.tobytes(), count=int(exported.shape[0]), interval=interval)

    def abandon(self, runtime, error):
        """Abandon the in-progress session without reloading the resident model."""
        from .conversation_control import SCHEMA
        self.record.append(dict(abandoned=True, error=type(error).__name__))
        if getattr(runtime, '_continuation_enabled', False) is not True:
            return
        try:
            _json(runtime.continuation_file, dict(conversation_control=dict(
                schema=SCHEMA, action='abandon',
                conversation_id=self.scenario.conversation_id,
                reason='local-sink-failure' if 'sink' in str(error).lower()
                       else 'local-preparation-failure')))
            runtime._generate(continuation=True)
        except BaseException:
            runtime._state = 'degraded'
        finally:
            try:
                _json(runtime.continuation_file, dict(mode='independent'))
            except BaseException:
                runtime._state = 'degraded'
