"""Isolated native VoxCPM2 candidate; no Irodori/default-route changes.

Evidence (public source inspected, not GPU acceptance):
https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/core.py
https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py
https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915
Source LICENSE and model card: Apache-2.0. Native AudioVAEV2 weights are
local audiovae.safetensors/audiovae.pth, not a separate Qwen/codec download.
LlamaTokenizerFast loads the local snapshot; MiniCPM4 is constructed locally.
Denoiser (ModelScope ZipEnhancer) and wetext normalization are disabled.
Upstream still collapses whitespace internally; adapter does not split text.

Deployment must stage/verify model bytes against MODEL_REVISION separately:
a directory name/config cannot establish weight provenance. Source installation
is checked using PEP610 metadata. Run with network egress disabled as defense
in depth. No weight downloads/installations are performed here.
"""
from importlib.metadata import distribution
import json
import hashlib
import dis
import inspect
import sys
import time
from pathlib import Path

MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
SOURCE_REVISION = "cce58a5b59303c9bd63d12f23afe6a49a4a80c59"
SOURCE_ARCHIVE_SHA256 = "a2b9d9f3076d9d6eaaf5fdb6b92e1bc7c3b4643bc9a489bace7622e04bcdbd2c"
# Same explicit codec/runtime pins as hf_conversation_tts; not installed here.
RUNTIME_PINS = ("torch==2.8.0", "torchaudio==2.8.0", "torchcodec==0.7.0",
                "transformers==5.15.0", "huggingface_hub==1.33.0")
MAX_NEW_TOKENS = 1024  # upstream calls this max_len, not max_new_tokens


SOURCE_TREE_SHA256 = '6df6c47053cb140e2400c8000786aa4127ce5a4964f7265eff2b321862768fcd'


def source_tree_sha(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*.py')):
        if p.is_symlink():raise RuntimeError('source symlink')
        h.update((str(p.relative_to(root))+'\0'+hashlib.sha256(p.read_bytes()).hexdigest()+'\n').encode())
    return h.hexdigest()


def _check_source():
    dist = distribution("voxcpm")
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw or "{}")
    url = info.get("url", "")
    git_ok = (url.removesuffix(".git") == "https://github.com/OpenBMB/VoxCPM"
              and info.get("vcs_info", {}).get("commit_id") == SOURCE_REVISION)
    archive_ok = (url in ("https://codeload.github.com/OpenBMB/VoxCPM/tar.gz/" + SOURCE_REVISION,
                           "https://github.com/OpenBMB/VoxCPM/archive/" + SOURCE_REVISION + ".tar.gz")
                  and info.get("archive_info", {}).get("hashes", {}).get("sha256") == SOURCE_ARCHIVE_SHA256)
    # uv's PEP610 archive_info may be empty even after --require-hashes.
    # Verify all38 installed Python files against the hash-checked archive,
    # rather than weakening immutable-source proof or trusting URL alone.
    if not git_ok and not archive_ok and url in (
            'https://github.com/OpenBMB/VoxCPM/archive/'+SOURCE_REVISION+'.tar.gz',
            'https://codeload.github.com/OpenBMB/VoxCPM/tar.gz/'+SOURCE_REVISION):
        archive_ok = source_tree_sha(dist.locate_file('voxcpm')) == SOURCE_TREE_SHA256
    if not (git_ok or archive_ok):
        raise RuntimeError("VoxCPM must be installed from the pinned upstream source")


def _stop_probe(model):
    """Locate the pinned Python stop branch, without changing installed code.

    The stop head is outside the compiled submodules. A line event on its
    taken break observes the *actual* decision, without a second tensor read.
    """
    method = getattr(model.tts_model, '_inference', None)
    if method is None:
        return None
    fn = inspect.unwrap(method)
    path = Path(inspect.getsourcefile(fn))
    if hashlib.sha256(path.read_bytes()).hexdigest() != '31ad554bc15a4da18cf820c6fda97a5b2c56ac8e9a5e66031a264d363661575c':
        raise RuntimeError('unknown VoxCPM2 stop source')
    lines, start = inspect.getsourcelines(fn)
    decision = next(i for i, line in enumerate(lines)
                    if line.strip() == 'if i > min_len and stop_flag == 1:')
    if lines[decision + 1].strip() != 'break':
        raise RuntimeError('unknown VoxCPM2 stop branch')
    return fn.__code__, start + decision + 1


def _verify_optimized(model):
    # Upstream optimize() catches exceptions, including partially applied
    # compilation. Check all four requested wrappers after its warm-up.
    import torch
    tts = model.tts_model
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError('compile fallback suppression must be disabled')
    steps = (tts.base_lm.forward_step, tts.residual_lm.forward_step)
    modules = (tts.feat_encoder, tts.feat_decoder.estimator)
    if (not all(callable(getattr(step, '_torchdynamo_orig_callable', None)) for step in steps)
            or not all(getattr(module, '_orig_mod', None) is not None for module in modules)):
        raise RuntimeError('requested VoxCPM2 compile activation failed or was partial')


class _Candidate:
    def __init__(self, model, reference, *, observe_stop=True):
        if type(observe_stop) is not bool:
            raise TypeError('observe_stop must be an explicit bool')
        self.observe_stop = observe_stop
        self._model = model
        self._reference = reference
        self.completion = None
        self._active = False
        self.sample_rate = int(model.tts_model.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("invalid native sample rate")

    def stream(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nonempty full text required")
        if self._active:
            raise RuntimeError('concurrent candidate streams forbidden')
        self._active = True
        self.completion = dict(reason='unknown', natural_stop=False,
                               started_monotonic_ns=time.monotonic_ns(),
                               decision_monotonic_ns=None, ended_monotonic_ns=None,
                               effective_max_len=None, stop_index=None,
                               stream_exhausted=False)
        evidence = self.completion
        chunks = None
        probe = None
        observed_frame = None

        def trace(frame, event, arg):
            nonlocal observed_frame
            if probe is None or frame.f_code is not probe[0]:
                return None
            if frame.f_locals.get('self') is not self._model.tts_model:
                return None
            if observed_frame is not None and observed_frame is not frame:
                raise RuntimeError('multiple native inference attempts forbidden')
            observed_frame = frame
            if event == 'line' and frame.f_lineno == probe[1]:
                reason = ('natural_stop' if frame.f_locals['i'] < frame.f_locals['max_len'] - 1
                          else 'max_len')
                evidence.update(reason=reason, stop_index=frame.f_locals['i'],
                                effective_max_len=frame.f_locals['max_len'],
                                decision_monotonic_ns=time.monotonic_ns())
            elif (event == 'return' and dis.opname[frame.f_code.co_code[frame.f_lasti]] == 'RETURN_VALUE'
                  and evidence['decision_monotonic_ns'] is None):
                limit = frame.f_locals.get('max_len')
                index = frame.f_locals.get('i')
                if isinstance(limit, int) and limit > 0 and index == limit - 1:
                    evidence.update(reason='max_len', stop_index=index,
                                    effective_max_len=limit,
                                    decision_monotonic_ns=time.monotonic_ns())
            return trace

        try:
            probe = _stop_probe(self._model) if self.observe_stop else None
            chunks = self._model.generate_streaming(
                text=text, reference_wav_path=self._reference,
                max_len=MAX_NEW_TOKENS, normalize=False, denoise=False,
                retry_badcase=False,
            )
            while True:
                # Thread-local and only during native advancement: never leave
                # tracing installed while the consumer owns a yielded chunk.
                previous = sys.gettrace() if self.observe_stop else None
                if self.observe_stop and previous is not None:
                    raise RuntimeError('stop telemetry requires an untraced thread')
                try:
                    if self.observe_stop:
                        sys.settrace(trace)
                    chunk = next(chunks)
                except StopIteration:
                    evidence['stream_exhausted'] = True
                    evidence['natural_stop'] = evidence['reason'] == 'natural_stop'
                    break
                finally:
                    if self.observe_stop:
                        sys.settrace(previous)
                yield chunk
        except GeneratorExit:
            evidence.update(cancel_prior_reason=evidence['reason'], reason='cancelled', natural_stop=False)
            raise
        except BaseException:
            evidence.update(reason='error', natural_stop=False)
            raise
        finally:
            try:
                if chunks is not None:
                    chunks.close()
            except BaseException:
                evidence.update(reason='error', natural_stop=False)
                raise
            finally:
                observed_frame = None
                evidence['ended_monotonic_ns'] = time.monotonic_ns()
                self._active = False


def load(model_dir, reference_path, reference_transcript=None, *, optimize=False):
    """Load explicit CUDA native model; eager default, opt-in compile profile.

    optimize=True includes upstream compilation and synthetic warm-up. Wrapper
    checks prove activation, not GPU quality or absence of backend graph breaks.

    Transcript is intentionally unused: VoxCPM2 reference_wav_path requires no
    transcript. prompt_text + prompt_wav_path instead selects continuation mode.
    """
    if type(optimize) is not bool:
        raise TypeError('optimize must be an explicit bool')
    model_dir = Path(model_dir).expanduser().resolve(strict=True)
    reference = Path(reference_path).expanduser().resolve(strict=True)
    if not model_dir.is_dir() or not reference.is_file():
        raise ValueError("local model directory and reference file required")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architecture", "").lower() != "voxcpm2":
        raise ValueError("explicit VoxCPM2 architecture required")
    _check_source()
    from voxcpm import VoxCPM
    model = VoxCPM(voxcpm_model_path=str(model_dir),
                   zipenhancer_model_path=None, enable_denoiser=False,
                   optimize=optimize, device="cuda")
    if optimize:
        _verify_optimized(model)
    return _Candidate(model, str(reference))
