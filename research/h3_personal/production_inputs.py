"""Validate mounted production inputs; stage rows only, never models or jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from . import freeze_dataset as freeze
from research.hf_prepare_identity_media import check_selection

CODE_PINS = {
    'hf_prepare_identity_media.py': '8ae4d5729ac9ea7a6273e8ca3e1feec4acd0030a6ca719b2ccc0b3de2395d31f',
    'freeze_dataset.py': 'e8fbb5f9be9ccc68637e38a3ae517a5ab34a5f9ff1601bea38aa1e178e59cc02',
    'preprocess.py': '3c51ebde6529d584a0485037ff29c17112f47677e057f6f89f6eb64bb29c6206',
}
HELPER_PINS = {k.removesuffix('.py'): v for k, v in CODE_PINS.items()
               if k != 'hf_prepare_identity_media.py'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _inside(root, name):
    _require(isinstance(name, str) and bool(name) and '\\' not in name,
             'relative POSIX file path required')
    parts = name.split('/')
    _require(not PurePosixPath(name).is_absolute() and
             all(p not in ('', '.', '..') for p in parts), 'unsafe relative path')
    root = Path(root).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file(), 'file escapes root or is not regular')
    return path


def _approval(mapping, schema):
    _require(isinstance(mapping, dict) and mapping.get('schema') == schema and
             mapping.get('approved') is True, 'explicit approved mapping required')
    _require(isinstance(mapping.get('approved_by'), str) and mapping['approved_by'].strip(),
             'mapping approver required')
    _require(isinstance(mapping.get('provenance'), dict) and mapping['provenance'] and
             isinstance(mapping['provenance'].get('source'), str) and
             mapping['provenance']['source'].strip(), 'mapping source provenance required')


def stage(*, selection_path, manifest_path, summary_path, source_map_path,
          reference_mapping_path, caption_mapping_path, source_root, media_root,
          reference_root, code_root, expected_input_revision):
    """Return (make_plan-compatible rows, receipt) after ALL checks pass.

    expected_input_revision must come from the operator's independent pin, not
    from summary.json. code_root contains the three flat mounted producer files.
    Original paths in the freeze are labels only: source_map relocates by digest.
    Hash each resolved file at most once during this invocation.
    """
    _require(isinstance(expected_input_revision, str) and
             re.fullmatch(r'[0-9a-f]{40}', expected_input_revision), 'input revision pin required')
    hashes = {}

    def sha(path):
        path = Path(path).resolve(strict=True)
        _require(path.is_file(), 'regular file required')
        if path not in hashes:
            hashes[path] = freeze.file_sha256(path)
        return hashes[path]

    def verify(path, digest):
        _require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
                 'SHA256 required')
        _require(sha(path) == digest, 'file byte hash mismatch')

    paths = dict(selection=selection_path, manifest=manifest_path, summary=summary_path,
                 source_map=source_map_path, references=reference_mapping_path,
                 captions=caption_mapping_path)
    documents = {key: _load(path) for key, path in paths.items()}
    selection, manifest, summary = (documents[k] for k in ('selection', 'manifest', 'summary'))
    frozen = check_selection(selection, freeze)
    _require(summary.get('complete_raw_preprocessing') is True and
             type(summary.get('processed_clips')) is int and summary['processed_clips'] == 715,
             'complete 715-clip summary required')
    _require(summary.get('input_revision') == expected_input_revision, 'input revision mismatch')
    _require(summary.get('freeze_sha256') == sha(selection_path), 'freeze byte pin mismatch')
    _require(summary.get('code_sha256') == CODE_PINS and
             summary.get('helper_sha256') == HELPER_PINS, 'producer code pins mismatch')
    for name, digest in CODE_PINS.items():
        verify(_inside(code_root, name), digest)
    dataset = selection['dataset_sha256']
    _require(manifest.get('schema') == 'h3-production-raw-media-v1' and
             manifest.get('dataset_sha256') == dataset and summary.get('dataset_sha256') == dataset,
             'raw schema/dataset mismatch')
    raw = manifest.get('samples')
    _require(isinstance(raw, list) and len(raw) == 715, '715 raw samples required')
    source_map = documents['source_map']
    _require(isinstance(source_map, dict) and
             set(source_map) == {r['source_sha256'] for r in frozen}, 'exact six-source map required')
    for digest, name in source_map.items():
        path = _inside(source_root, name)
        _require(path.suffix.lower() == '.mov', 'original MOV required')
        verify(path, digest)

    refs, captions = documents['references'], documents['captions']
    _approval(refs, 'h3-production-reference-mapping-v1')
    _approval(captions, 'h3-production-caption-mapping-v1')
    ids = {r['id'] for r in frozen}
    associations = refs.get('samples')
    _require(isinstance(associations, dict) and set(associations) == ids,
             'reference association must cover exactly 715 IDs')
    scope, values = captions.get('scope'), captions.get('captions')
    _require(scope in ('sample', 'session'), 'explicit caption scope required')
    expected = ids if scope == 'sample' else {r['session_id'] for r in frozen}
    _require(isinstance(values, dict) and set(values) == expected and
             all(isinstance(v, str) and v.strip() for v in values.values()),
             'explicit nonempty captions must cover exact scope')
    rows = []
    for selected, record in zip(frozen, raw):
        _require(isinstance(record, dict) and
                 all(record.get(k) == v for k, v in selected.items()),
                 'ordered frozen ID/source/range/pool mismatch')
        proof = {k: selected[k] for k in ('source_sha256', 'start_seconds', 'end_seconds')}
        _require(record.get('input_provenance') == proof, 'raw source/range proof mismatch')
        sid = selected['id']
        files = {}
        for kind, directory, suffix in (('video', 'clips', '.mp4'), ('audio', 'audio', '.wav')):
            _require(record.get(kind) == directory + '/' + sid + suffix, 'raw path mismatch')
            path = _inside(media_root, record[kind])
            verify(path, record.get(kind + '_sha256'))
            files[kind] = str(path)
        ref = associations[sid]
        _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'},
                 'one explicit reference path/hash per sample required')
        image = _inside(reference_root, ref['path'])
        verify(image, ref['sha256'])
        caption_key = sid if scope == 'sample' else selected['session_id']
        rows.append(dict(id=sid, **files, reference_image=str(image), caption=values[caption_key],
                         quality_pool=selected['quality_pool'],
                         **{k: selection['geometry'][k] for k in ('width', 'height', 'num_frames')}))
    receipt = {
        'schema': 'h3-production-inputs-receipt-v1', 'dataset_sha256': dataset,
        'sample_count': 715, 'pool_counts': selection['pool_counts'],
        'input_revision': expected_input_revision, 'code_sha256': CODE_PINS,
        'input_documents': {k: {'path': str(Path(p).resolve()), 'sha256': sha(p)} for k, p in paths.items()},
        'reference_mapping': refs, 'caption_mapping': captions,
        'rows_sha256': hashlib.sha256(freeze.canonical(rows)).hexdigest(),
        'verified_file_count': len(hashes), 'mounted_bytes_verified': True,
        'authenticated_provenance': False, 'legacy_cache_certified': False,
        'technical_training_ready': False, 'c_gate_enabled': False,
    }
    return rows, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'manifest', 'summary', 'source-map', 'reference-mapping',
                 'caption-mapping', 'source-root', 'media-root', 'reference-root', 'code-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-input-revision', required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    for name in ('selection', 'manifest', 'summary', 'source_map', 'reference_mapping', 'caption_mapping'):
        args[name + '_path'] = args.pop(name)
    rows, receipt = stage(**args)
    # Exclusive directory: never overwrite an earlier stage or publish invalid rows.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rows.json').write_bytes(freeze.canonical(rows) + b'\n')
    (output / 'receipt.json').write_bytes(freeze.canonical(receipt) + b'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
