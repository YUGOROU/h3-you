"""Provider-neutral canonical checkpoint escrow and weight-only handoff.

No optimizer/counter/RNG resume, downloads or quality claims. Immutable remote
revision resolution and publication transport belong to the bounded worker.
"""
import hashlib
import json
from pathlib import Path
import re
from . import identity_checkpoint as cp

SCHEMA = "h3-identity-weight-handoff-v1"
ARTIFACTS = (cp.WEIGHTS, cp.CONFIG, cp.RECEIPT, cp.EXPORTED)


def recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_sampling_cursor(cursor, *, minimum_draws=0):
    if (not isinstance(cursor, dict)
            or set(cursor) != {'consumed_draws', 'mode', 'epoch_budget', 'seed', 'index_sha256'}
            or type(cursor['consumed_draws']) is not int or cursor['consumed_draws'] < minimum_draws
            or cursor['mode'] not in ('replacement', 'coverage_first')
            or type(cursor['epoch_budget']) is not int or cursor['epoch_budget'] != 715
            or type(cursor['seed']) is not int or not 0 <= cursor['seed'] < 2**63
            or cursor['seed'] + cursor['consumed_draws'] // 715 >= 2**63
            or not isinstance(cursor['index_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', cursor['index_sha256'])):
        raise ValueError('invalid deterministic sampling cursor')
    return cursor


def escrow_metadata(checkpoint, *, pins, recipe_sha256, run_id, step, sampling_cursor=None):
    """Seal exported-only tensors; deliberately NOT a fresh-base verification."""
    cp.inspect_export(checkpoint, expected_pins=pins)
    if (not isinstance(recipe_sha256, str) or not re.fullmatch('[0-9a-f]{64}', recipe_sha256)
            or not isinstance(run_id, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id)
            or type(step) is not int or step <= 0):
        raise ValueError('explicit recipe/run/update binding required')
    if sampling_cursor is not None:
        validate_sampling_cursor(sampling_cursor, minimum_draws=step)
    return dict(schema=SCHEMA, status='exported_only_not_fresh_base_verified',
                parent_run_id=run_id, parent_step=step, recipe_sha256=recipe_sha256,
                artifacts={name: cp.sha256(Path(checkpoint)/name) for name in ARTIFACTS},
                optimizer_state_present=False, exact_resume=False, quality_verified=False,
                sampling_cursor=sampling_cursor)


def validate_handoff(checkpoint, metadata, *, pins, expected_recipe_sha256):
    """Verify all source bytes/recipe before fresh target model is mutated."""
    cp.inspect_export(checkpoint, expected_pins=pins)
    if (not isinstance(expected_recipe_sha256, str)
            or not re.fullmatch('[0-9a-f]{64}', expected_recipe_sha256)):
        raise ValueError('explicit expected recipe digest required')
    if (not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA
            or metadata.get('status') != 'exported_only_not_fresh_base_verified'
            or metadata.get('recipe_sha256') != expected_recipe_sha256
            or not isinstance(metadata.get('parent_run_id'), str)
            or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', metadata['parent_run_id'])
            or type(metadata.get('parent_step')) is not int or metadata['parent_step'] <= 0
            or any(metadata.get(k) is not False for k in ('optimizer_state_present','exact_resume','quality_verified'))
            or not isinstance(metadata.get('artifacts'), dict)
            or set(metadata['artifacts']) != set(ARTIFACTS)):
        raise ValueError('invalid weight-only handoff metadata')
    if metadata.get('sampling_cursor') is not None:
        validate_sampling_cursor(metadata['sampling_cursor'], minimum_draws=metadata['parent_step'])
    for name, digest in metadata['artifacts'].items():
        if cp.sha256(Path(checkpoint)/name) != digest:
            raise ValueError('handoff artifact digest mismatch')
    return metadata


def restore_weights(fresh_model, checkpoint, metadata, *, pins, expected_recipe_sha256, boundary=None):
    """Fresh-base strict reload; caller creates NEW optimizer/counters/RNG."""
    validate_handoff(checkpoint, metadata, pins=pins, expected_recipe_sha256=expected_recipe_sha256)
    cp.strict_reload(fresh_model, checkpoint, freshly_constructed=True, expected_pins=pins, boundary=boundary)
    return dict(schema=SCHEMA, status='weights_loaded_into_fresh_base',
                parent_run_id=metadata['parent_run_id'], parent_step=metadata['parent_step'],
                parent_artifacts=metadata['artifacts'], recipe_sha256=expected_recipe_sha256,
                new_optimizer=True, new_counters=True, exact_resume=False, quality_verified=False,
                sampling_cursor=metadata.get('sampling_cursor'))
