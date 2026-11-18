"""Experimental H3 block-boundary CP plan, not a GPU launcher or approval.

Replicate modality packing/text refinement; split the block-stack packed sequence
only. Gather before global output normalization and modality index selection.
GPU parity, performance and identity are unverified. Production remains unchanged.
"""
from hashlib import sha256
from pathlib import Path

DIFFUSERS_REVISION = 'abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc'


def block_plan(*, layers, sequence_rows, heads, world_size=2, padding_rows=0):
    for name, value in [('layers', layers), ('sequence_rows', sequence_rows),
                        ('heads', heads), ('world_size', world_size), ('padding_rows', padding_rows)]:
        if type(value) is not int or value < (0 if name == 'padding_rows' else 1):
            raise ValueError(f'invalid {name}')
    if world_size != 2 or layers < 2:
        raise ValueError('experimental two-rank multi-block plan only')
    if padding_rows or sequence_rows % world_size or heads % world_size:
        raise ValueError('padless divisible sequence and head counts required; no silent padding')
    return {
        'transformer_blocks.*': {
            'adaln_indices': {'split_dim': 0, 'expected_dims': 1},
            'rotary_emb': [{'split_dim': 0, 'expected_dims': 2} for _ in range(2)],
        },
        'transformer_blocks.0': {'hidden_states': {'split_dim': 1, 'expected_dims': 3}},
        f'transformer_blocks.{layers-1}': {'gather_dim': 1, 'expected_dims': 3},
    }


def materialize_plan(descriptors):
    """Convert audited descriptors with the pinned Diffusers API, without enabling it.

    The installer must disable CP on replicated token_refiner attention processors,
    choose a verified CP-capable backend, assert local tensor shapes and run numeric
    parity first. Calling generic enable_parallelism alone is insufficient.
    """
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
    def inputs(value):
        if isinstance(value, list):
            return [inputs(v) for v in value]
        return ContextParallelInput(**value)
    return {name: (ContextParallelOutput(**spec) if 'gather_dim' in spec else
                   {key: inputs(value) for key, value in spec.items()})
            for name, spec in descriptors.items()}


def verify_source_files(root, receipts):
    """Require exact bytes for every audited source; never accept HEAD/main drift."""
    if not receipts:
        raise ValueError('source receipts required')
    for entry in receipts:
        if entry.get('revision') != DIFFUSERS_REVISION:
            raise ValueError('Diffusers revision differs')
        relative = Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe source path')
        base = Path(root).resolve(strict=True)
        path = base / relative
        descendants = [base.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts)+1)]
        if any(p.is_symlink() for p in descendants) or not path.is_file():
            raise ValueError('regular audited source required')
        if sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('audited source bytes differ')
