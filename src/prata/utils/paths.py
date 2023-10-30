from pathlib import Path
import pickle
import hashlib

CACHE_PATH = Path("~/.cache/trungdt21")
CACHE_PATH.mkdir(exist_ok=True, parents=True)


def generate_cachepath_from_path(path: Path):
    CACHE_PATH_DIR = CACHE_PATH / "filelists"
    CACHE_PATH_DIR.mkdir(exist_ok=True, parents=True)
    path = path.resolve().as_posix()
    md5sum = hashlib.md5(path.encode("utf-8")).hexdigest()
    return CACHE_PATH_DIR / md5sum


def get_filelist_and_cache(path: Path, glob_str: str):
    CACHE_PATH_DIR = CACHE_PATH / "filelists"
    CACHE_PATH_DIR.mkdir(exist_ok=True, parents=True)
    pathstr = path.resolve().as_posix()
    md5sum = hashlib.md5(pathstr.encode("utf-8")).hexdigest()
    cachepath = CACHE_PATH_DIR / f"{md5sum}_{glob_str.replace('*','')}.pkl"
    reload = True
    filelist = []
    if cachepath.exists():
        k = input(f"Found filelist cache for {pathstr}, load it? (y/n)")
        if k.lower() and k == "y":
            with open(cachepath, "rb") as f:
                filelist = pickle.load(f)
                reload = False
        else:
            reload = True
    if reload:
        filelist = list(path.rglob(glob_str))
        with open(cachepath, "wb") as f:
            pickle.dump(filelist, f)
    return filelist
