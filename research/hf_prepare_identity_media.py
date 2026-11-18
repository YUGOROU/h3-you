"""Bounded CPU raw-media preparation. No models, training, or job submission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

HELPER_SHA256 = {
    'freeze_dataset': 'e8fbb5f9be9ccc68637e38a3ae517a5ab34a5f9ff1601bea38aa1e178e59cc02',
    'preprocess': '3c51ebde6529d584a0485037ff29c17112f47677e057f6f89f6eb64bb29c6206',
}
GLOBAL_SECONDS = 10500  # below the platform's hard 3h, including setup allowance
PROCESS_SECONDS = 180
SNAPSHOT_CLIPS = 10
SNAPSHOT_SECONDS = 300


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def helpers(directory):
    modules = []
    for name, digest in HELPER_SHA256.items():
        path = Path(directory) / (name + '.py')
        if sha256(path) != digest:
            raise ValueError('untrusted helper bytes: ' + name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def check_selection(selection, freeze):
    if selection.get('schema') != 'h3-production-selection-v1':
        raise ValueError('unsupported frozen selection')
    if selection.get('selection_frozen') is not True or selection.get('source_bytes_verified') is not True:
        raise ValueError('verified immutable freeze required')
    rows = selection['samples']
    keys = {'id', 'source', 'source_sha256', 'session_id', 'start_seconds', 'end_seconds', 'quality_pool'}
    if any(set(row) != keys for row in rows):
        raise ValueError('unexpected frozen row fields')
    rebuilt = freeze.build([dict(r, pool=r['quality_pool'], human_review='approved_sampled_images')
                            for r in rows], verify_sources=False)
    for key in ('dataset_sha256', 'sample_count', 'pool_counts', 'geometry'):
        if selection.get(key) != rebuilt[key]:
            raise ValueError('frozen semantic mismatch: ' + key)
    seconds = selection.get('clip_seconds_with_tail_overlap')
    if (type(seconds) not in (int, float) or not math.isfinite(seconds)
            or abs(seconds - 3694.166667) > 1e-5
            or abs(seconds - rebuilt['clip_seconds_with_tail_overlap']) > 1e-5):
        raise ValueError('frozen duration mismatch')
    if len({r['source_sha256'] for r in rows}) != 6:
        raise ValueError('exactly six known originals required')
    return rows


def verify_sources(rows, source_map):
    expected = {r['source_sha256'] for r in rows}
    if set(source_map) != expected:
        raise ValueError('source map must contain exactly the frozen originals')
    for digest, filename in source_map.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != '.mov':
            raise ValueError('regular original MOV required')
        if sha256(path) != digest:
            raise ValueError('source bytes changed')


def run_bounded(command, timeout):
    """Kill the entire validation/FFmpeg process group on timeout or global alarm."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def encode(row, source, output, preprocess, helper_dir, timeout=PROCESS_SECONDS):
    """Exactly one FFmpeg encoder at a time; validation also has a subprocess timeout."""
    started = time.monotonic()
    video = output / 'clips' / (row['id'] + '.mp4')
    audio = output / 'audio' / (row['id'] + '.wav')
    pv = video.with_suffix('.partial.mp4')
    pa = audio.with_suffix('.partial.wav')
    try:
        run_bounded(preprocess.command(str(source), row['start_seconds'], pv, pa, 768, 1344), timeout)
        validate_files(pv, pa, helper_dir, timeout)
        pv.replace(video)
        pa.replace(audio)
    finally:
        pv.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)
    return dict(row, video='clips/' + video.name, audio='audio/' + audio.name,
                video_sha256=sha256(video), audio_sha256=sha256(audio),
                input_provenance={'source_sha256': row['source_sha256'],
                                  'start_seconds': row['start_seconds'], 'end_seconds': row['end_seconds']},
                preparation_seconds=time.monotonic() - started)


def validate_files(video, audio, helper_dir, timeout=PROCESS_SECONDS):
    # Existing validate() includes ffprobe + full video decode. Enclosing it in a
    # child process bounds both helpers without modifying/monkeypatching them.
    script = ('import sys; sys.path.insert(0,sys.argv[1]); from pathlib import Path; '
              'from preprocess import validate; validate(Path(sys.argv[2]),Path(sys.argv[3]),768,1344)')
    run_bounded([os.sys.executable, '-c', script, str(helper_dir), str(video), str(audio)], timeout)


class PrivateSnapshots:
    def __init__(self, api, repo, output, dataset_sha256, input_revision):
        self.api, self.repo, self.output = api, repo, Path(output)
        self.prefix = 'raw/' + dataset_sha256
        self.input_revision = input_revision

    def private(self):
        if self.api.repo_info(self.repo, repo_type='dataset').private is not True:
            raise ValueError('private dataset required')

    def upload(self, records, dirty):
        from huggingface_hub import CommitOperationAdd
        self.private()
        names = ['manifest.json', 'summary.json']
        for row in dirty:
            names.extend((row['video'], row['audio']))
        self.api.create_commit(repo_id=self.repo, repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=self.prefix + '/' + n,
                                           path_or_fileobj=str(self.output / n)) for n in names],
            commit_message='Bounded raw-media snapshot')

    def restore(self, rows):
        """Explicit resume only; a missing/broken snapshot is an error, not a fresh retry."""
        from huggingface_hub import hf_hub_download
        import shutil
        self.private()
        revision = self.api.repo_info(self.repo, repo_type='dataset').sha
        def fetch(name):
            return Path(hf_hub_download(self.repo, self.prefix + '/' + name,
                                       repo_type='dataset', revision=revision))
        manifest = json.loads(fetch('manifest.json').read_text())
        summary = json.loads(fetch('summary.json').read_text())
        if summary['input_revision'] != self.input_revision:
            raise ValueError('resume input revision changed')
        if len(manifest['samples']) > len(rows):
            raise ValueError('unexpected resume count')
        # Validate the entire ordered selection before downloading any media.
        for frozen, record in zip(rows, manifest['samples']):
            if any(record.get(k) != v for k, v in frozen.items()):
                raise ValueError('resume selection changed')
        for row in manifest['samples']:
            ident = row['id']
            if not isinstance(ident, str) or not __import__('re').fullmatch(r'[A-Za-z0-9_.-]+', ident) or ident in ('.', '..'):
                raise ValueError('unsafe resume ID')
            for key, directory, extension in (('video', 'clips', '.mp4'), ('audio', 'audio', '.wav')):
                name = directory + '/' + ident + extension
                if row[key] != name:
                    raise ValueError('unsafe resume path')
                shutil.copyfile(fetch(name), self.output / name)
        dump(self.output / 'manifest.json', manifest)
        return manifest


def prepare(request, inputs, output, snapshots, resume=False, encoder=encode):
    started = time.monotonic()
    freeze, preprocess = helpers(inputs)
    if sha256(Path(inputs) / 'frozen_selection.json') != request['freeze_sha256']:
        raise ValueError('freeze byte hash mismatch')
    selection = json.loads((Path(inputs) / 'frozen_selection.json').read_text())
    rows = check_selection(selection, freeze)
    if request['dataset_sha256'] != selection['dataset_sha256']:
        raise ValueError('request semantic mismatch')
    verify_sources(rows, request['source_map'])  # ALL sources, before any encoding/resume
    output = Path(output)
    if any(output.resolve() == Path(p).resolve().parent or Path(p).resolve().parent in output.resolve().parents
           for p in request['source_map'].values()):
        raise ValueError('output must be separate from originals')
    if output.exists() and not resume and any(output.iterdir()):
        raise ValueError('nonempty output requires explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('clips', 'audio'):
        (output / name).mkdir(exist_ok=True)
    records = []
    manifest = {'schema': 'h3-production-raw-media-v1', 'dataset_sha256': selection['dataset_sha256'],
                'samples': records}
    if resume:
        restored = snapshots.restore(rows)
        if restored['dataset_sha256'] != selection['dataset_sha256'] or len(restored['samples']) > len(rows):
            raise ValueError('resume selection mismatch')
        for frozen, record in zip(rows, restored['samples']):
            if any(record.get(k) != v for k, v in frozen.items()):
                raise ValueError('resume provenance mismatch')
            if (record['video'] != 'clips/' + frozen['id'] + '.mp4'
                    or record['audio'] != 'audio/' + frozen['id'] + '.wav'):
                raise ValueError('unsafe resume path')
            for key in ('video', 'audio'):
                if sha256(output / record[key]) != record[key + '_sha256']:
                    raise ValueError('resume bytes changed')
            validate_files(output / record['video'], output / record['audio'], inputs)
            records.append(record)
    summary = {'dataset_sha256': selection['dataset_sha256'], 'input_revision': request['input_revision'],
               'freeze_sha256': request['freeze_sha256'], 'helper_sha256': HELPER_SHA256,
               'code_sha256': request['code_sha256'],
               'complete_raw_preprocessing': False, 'processed_clips': len(records),
               'technical_training_ready': False, 'h3_used': False, 'encoder_concurrency': 1}
    dirty = []
    last = time.monotonic()
    def checkpoint(complete=False):
        nonlocal last
        summary.update(processed_clips=len(records), complete_raw_preprocessing=complete,
                       elapsed_seconds=time.monotonic() - started)
        dump(output / 'manifest.json', manifest)
        dump(output / 'summary.json', summary)
        snapshots.upload(records, dirty)
        dirty.clear()
        last = time.monotonic()
    checkpoint()
    try:
        for row in rows[len(records):]:
            record = encoder(row, request['source_map'][row['source_sha256']], output, preprocess, inputs)
            records.append(record)
            dirty.append(record)
            if len(dirty) >= SNAPSHOT_CLIPS or time.monotonic() - last >= SNAPSHOT_SECONDS:
                checkpoint()
        if len(records) != 715:
            raise ValueError('incomplete selection')
        checkpoint(complete=True)
    except BaseException as error:
        # No stderr/exception messages: SDK errors may contain credentials or local paths.
        summary['failure_type'] = type(error).__name__
        summary['complete_raw_preprocessing'] = False
        dump(output / 'summary.json', summary)
        dump(output / 'manifest.json', manifest)
        raise
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('/inputs'))
    p.add_argument('--output', type=Path, default=Path('/work/raw'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    def expired(*_):
        raise TimeoutError('global preparation deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(GLOBAL_SECONDS)
    try:
        request = json.loads((a.inputs / 'request.json').read_text())
        revision = os.environ.get('HF_INPUT_REVISION', '')
        if (request.get('schema') != 'h3-cpu-raw-request-v1'
                or not __import__('re').fullmatch(r'[0-9a-f]{40}', revision)):
            raise ValueError('pinned input revision required')
        request['input_revision'] = revision
        expected_map = {d: str(a.inputs / 'sources' / (d + '.mov')) for d in request['source_map']}
        if request['source_map'] != expected_map:
            raise ValueError('unexpected mounted source paths')
        for name, digest in request['code_sha256'].items():
            if name not in ('hf_prepare_identity_media.py', 'freeze_dataset.py', 'preprocess.py') or sha256(a.inputs / name) != digest:
                raise ValueError('worker code hash mismatch')
        if set(request['code_sha256']) != {'hf_prepare_identity_media.py', 'freeze_dataset.py', 'preprocess.py'}:
            raise ValueError('missing code pins')
        from huggingface_hub import HfApi
        api = HfApi()
        sink = PrivateSnapshots(api, request['repo'], a.output, request['dataset_sha256'], request['input_revision'])
        sink.private()
        result = prepare(request, a.inputs, a.output, sink, resume=a.resume)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        print(json.dumps({'state': 'failed_closed', 'failure_type': type(error).__name__}), flush=True)
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
