"""Local, stdlib-only component reuse planning; never executes preprocessing.

Inspected worker: hf_train_personal.py uses the pinned Trainer's process_dataset
with --references --resolution-bucket 768x1344x124. Its index.json (version 1,
geometry, reference_canvas, samples) has no source/recipe hashes. Tensor folders
are latents, audio_latents, reference_latents and conditions. Existence, caption,
row counts, and source paths alone are NOT reuse evidence.

Contract: settings explicitly supplies geometry, encode_audio, references,
reference_canvas, lora_trigger, keyframes (empty only), and recipes for each
component. Each recipe requires immutable model/trainer/diffusers revisions and
a nonempty `recipe` object declaring the complete encoder/decode/processor
configuration (including dtype, tiling, normalization, seeds, library versions).
No defaults are inferred from the worker. Missing settings abort planning.

Optional local provenance.json: {"schema_version": 1, "artifacts": [
 {"component": "video", "fingerprint": "<sha256>",
  "path": "latents/example.safetensors", "sha256": "<artifact sha256>"}]}
Paths are relative to cache_root and may not escape it. Fingerprints must have
been recorded by a trusted producer *at encoding time*, not retrofitted from
current inputs. A plan is not a provenance attestation or a tensor validation:
only byte hashes are checked; shapes, numerical equivalence, loader integration,
and actual prior cache compatibility remain unverified. Legacy index.json is
never promoted to provenance. Work units deduplicate identical components; their
shards are disjoint assignments, not executable Trainer manifest shards. Text
must follow media/reference processing. No files are copied or cache overwritten.

CLI: python -m research.h3_personal.cache_plan --manifest manifest.json
     --settings settings.json [--cache-root .precomputed] [--shards 2]
JSON is printed to stdout only. Paths in a manifest resolve relative to that file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

COMPONENTS = ('video', 'audio', 'reference', 'text')
_HEX = re.compile(r'^[0-9a-f]{64}$')
_PIN = re.compile(r'^[0-9a-f]{40}$')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    """Stream bytes, not mtime, filename, or a decoded-media approximation."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_settings(settings):
    required = {'geometry', 'encode_audio', 'references', 'reference_canvas',
                'lora_trigger', 'keyframes', 'recipes'}
    if not isinstance(settings, dict) or not required <= settings.keys():
        raise ValueError('Missing explicit preprocessing settings')
    geometry = settings['geometry']
    if not isinstance(geometry, dict) or not {'width', 'height', 'num_frames', 'fps'} <= geometry.keys():
        raise ValueError('Missing geometry/clock')
    for key in ('width', 'height', 'num_frames', 'fps'):
        if type(geometry[key]) is not int or geometry[key] <= 0:
            raise ValueError('Geometry/clock must be positive integers')
    if any(type(settings[key]) is not bool for key in ('encode_audio', 'references')):
        raise ValueError('Audio/reference switches must be booleans')
    if settings['reference_canvas'] not in ('native', 'target'):
        raise ValueError('Unknown reference canvas')
    if settings['keyframes'] != []:
        raise ValueError('Keyframe caching is not supported; refusing incomplete fingerprints')
    if settings['lora_trigger'] is not None and not isinstance(settings['lora_trigger'], str):
        raise ValueError('Invalid LoRA trigger')
    recipes = settings['recipes']
    if not isinstance(recipes, dict):
        raise ValueError('Missing recipes')
    for component in COMPONENTS:
        recipe = recipes.get(component)
        if not isinstance(recipe, dict):
            raise ValueError(f'Missing {component} recipe provenance')
        for key in ('model_revision', 'trainer_revision', 'diffusers_revision'):
            if not isinstance(recipe.get(key), str) or not _PIN.fullmatch(recipe[key]):
                raise ValueError(f'{component}: {key} must be an immutable 40-hex commit')
        if not isinstance(recipe.get('recipe'), dict) or not recipe['recipe']:
            raise ValueError(f'{component}: missing encoder/processor recipe')
    _digest(settings)  # Reject non-JSON values and nonfinite numbers.


def component_fingerprints(row, settings, base_dir=Path('.')):
    """Content-addressed keys; reference presentation also affects text."""
    _validate_settings(settings)
    if not isinstance(row, dict) or not isinstance(row.get('caption'), str):
        raise ValueError('Each row requires an explicit string caption')
    # Canonical worker columns only: do not silently ignore unsupported aliases.
    allowed = {'id', 'video', 'audio', 'caption', 'reference_image',
               'reference_video', 'reference_audio'}
    if set(row) - allowed:
        raise ValueError(f'Unsupported manifest fields: {sorted(set(row) - allowed)}')
    base_dir = Path(base_dir)

    def media(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Media paths must be nonempty strings')
        path = Path(value)
        return file_sha256(path if path.is_absolute() else base_dir / path)

    video = media(row.get('video'))
    geometry = settings['geometry']
    refs = []
    if settings['references']:
        for kind in ('image', 'video', 'audio'):
            values = row.get('reference_' + kind)
            if values is None or values == '' or values == []:
                continue
            for value in values if isinstance(values, list) else [values]:
                refs.append({'kind': kind, 'sha256': media(value)})
    # Explicit WAV decouples target audio from target video. Embedded audio does not.
    audio = None
    if settings['encode_audio']:
        audio = {'sha256': media(row['audio']), 'source': 'explicit'} if row.get('audio') else {
            'sha256': video, 'source': 'embedded'}
    caption = row['caption']
    if settings['lora_trigger']:
        caption = f"{settings['lora_trigger']} {caption}".strip()
    dependencies = {
        'video': {'source': video, 'geometry': geometry},
        'audio': {'source': audio, 'num_frames': geometry['num_frames'], 'fps': geometry['fps']},
        'reference': {'media': refs, 'canvas': settings['reference_canvas'],
                      # Images use native sizing; video/audio depend on target duration.
                      'geometry': geometry if any(r['kind'] in ('video', 'audio') for r in refs) else None},
        'text': {'caption': caption, 'references': refs,
                 'geometry': geometry if any(r['kind'] in ('video', 'audio') for r in refs) else None,
                 'reference_canvas': settings['reference_canvas'] if refs else None,
                 'reference_recipe': settings['recipes']['reference'] if refs else None},
    }
    result = {}
    for component in COMPONENTS:
        if (component == 'audio' and audio is None) or (component == 'reference' and not refs):
            result[component] = None
        else:
            result[component] = _digest({'schema_version': 1, 'component': component,
                                        'recipe': settings['recipes'][component],
                                        'dependencies': dependencies[component]})
    return result


def _prior_artifacts(cache_root):
    """Fail closed on absent/malformed sidecars or individual artifact evidence."""
    accepted, diagnostics = {}, []
    if cache_root is None:
        return accepted, ['No prior cache supplied']
    root = Path(cache_root).resolve()
    try:
        payload = json.loads((root / 'provenance.json').read_text())
        if type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
            raise ValueError('Unsupported provenance schema')
        entries = payload['artifacts']
        if not isinstance(entries, list):
            raise ValueError('Artifacts must be a list')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return accepted, [f'Prior cache rejected: missing/invalid provenance ({type(exc).__name__})']
    for entry in entries:
        try:
            component, fingerprint = entry['component'], entry['fingerprint']
            if component not in COMPONENTS or not _HEX.fullmatch(fingerprint):
                raise ValueError('Invalid component/fingerprint')
            if not _HEX.fullmatch(entry['sha256']):
                raise ValueError('Missing artifact hash')
            relative = Path(entry['path'])
            path = (root / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
                raise ValueError('Artifact escapes cache root')
            if not path.is_file() or path.stat().st_size == 0 or file_sha256(path) != entry['sha256']:
                raise ValueError('Artifact missing/empty/hash mismatch')
            key = (component, fingerprint)
            candidate = {'path': str(path), 'sha256': entry['sha256']}
            # Deterministic selection when multiple valid copies have the same key.
            if key not in accepted or candidate['path'] < accepted[key]['path']:
                accepted[key] = candidate
        except (OSError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append(f'Artifact rejected ({type(exc).__name__})')
    return accepted, diagnostics


def plan_cache_reuse(rows, settings, cache_root=None, shards=2, base_dir=Path('.')):
    """Return a read-only plan with unique component work split into <=2 shards."""
    if type(shards) is not int or shards not in (1, 2):
        raise ValueError('shards must be 1 or 2')
    _validate_settings(settings)
    if not isinstance(rows, list):
        raise ValueError('Manifest must be a list')
    ids = [row.get('id') if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', ident)
           or ident in ('.', '..') for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError('Explicit unique safe sample IDs are required')
    prior, diagnostics = _prior_artifacts(cache_root)
    samples, pending = [], {}
    for row in sorted(rows, key=lambda value: value['id']):
        decisions = {}
        for component, fingerprint in component_fingerprints(row, settings, base_dir).items():
            key = (component, fingerprint)
            decision = {'fingerprint': fingerprint}
            if fingerprint is None:
                decision['action'] = 'not_required'
            elif key in prior:
                decision.update(action='reuse', artifact=prior[key])
            else:
                decision.update(action='compute', reason='No matching verified local provenance')
                pending.setdefault(key, []).append(row['id'])
            decisions[component] = decision
        samples.append({'id': row['id'], 'components': decisions})
    work = [{'component': component, 'fingerprint': fingerprint, 'sample_ids': sample_ids}
            for (component, fingerprint), sample_ids in sorted(pending.items())]
    return {'schema_version': 1, 'local_only': True, 'tensor_validation_performed': False,
            'samples': samples, 'shards': [work[rank::shards] for rank in range(shards)],
            'diagnostics': sorted(diagnostics),
            'execution_order': ['video/audio/reference', 'text'],
            'limits': 'Byte/provenance checks only; no tensor loading or actual prior cache compatibility validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--shards', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    try:
        result = plan_cache_reuse(json.loads(args.manifest.read_text()),
                                  json.loads(args.settings.read_text()), args.cache_root,
                                  args.shards, args.manifest.resolve().parent)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
