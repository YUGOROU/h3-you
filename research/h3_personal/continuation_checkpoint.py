"""Private per-rank adapter copies: strict reload publishes markers to its path."""
from pathlib import Path
import shutil
from .identity_checkpoint import WEIGHTS, CONFIG, RECEIPT, EXPORTED, sha256


def rank_checkpoint(source, output, rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError('exact2rank checkpoint required')
    source=Path(source);output=Path(output)
    names=(WEIGHTS,CONFIG,RECEIPT,EXPORTED)
    before={}
    for name in names:
        p=source/name
        if not p.is_file() or p.is_symlink():
            raise ValueError('regular canonical checkpoint artifacts required')
        before[name]=sha256(p)
    destination=output/('strict-reload-rank-'+str(rank))
    destination.mkdir(mode=0o700)  # Exclusive: no stale/uncertain retry reuse.
    for name in names:
        shutil.copyfile(source/name,destination/name)
    if any(sha256(destination/n)!=h or sha256(source/n)!=h for n,h in before.items()):
        raise ValueError('rank-local checkpoint copy differs from canonical source')
    return destination
