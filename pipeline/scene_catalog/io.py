"""Atomic artifact writes, content identities, and exclusive run ownership."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,os,tempfile
import numpy as np

def digest_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

@contextmanager
def run_lock(directory: Path):
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError(f'Another process owns this output: {directory}') from exc
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
