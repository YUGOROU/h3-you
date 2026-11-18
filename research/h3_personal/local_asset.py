"""Bounded staging from remote mounts before distributed/model initialization."""
import errno
import hashlib
from pathlib import Path
import time


def stage(source,destination,expected,*,attempts=3):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary=destination.with_suffix(destination.suffix+'.partial')
    if temporary.exists() or temporary.is_symlink():raise FileExistsError(temporary)
    if type(attempts) is not int or not 1<=attempts<=3:raise ValueError('bounded IO attempts required')
    for attempt in range(attempts):
        try:
            digest=hashlib.sha256()
            with Path(source).open('rb') as src,temporary.open('xb') as dst:
                while block:=src.read(1024*1024):
                    dst.write(block);digest.update(block)
            if digest.hexdigest()!=expected:raise ValueError('local asset SHA mismatch')
            destination.hardlink_to(temporary)
            temporary.unlink()
            return destination
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            if exc.errno!=errno.EIO or attempt+1==attempts:raise
            time.sleep(attempt+1)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
