from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT = "r26fh"
LISTING = f"https://api.osf.io/v2/nodes/{PROJECT}/files/osfstorage/?page[size]=100"
CHUNK = 1 << 20

DESCRIPTION = """Download ANPHY-Sleep (Wei et al., 2024) from OSF project r26fh and unpack it.

The release has one zip of about 3 GB per subject (86 GB compressed, 132 GB
unpacked) plus the montage, readme, demographics, artifact matrices and code.
Downloads resume after an interruption, every file is checked against the md5
OSF publishes, and subjects already unpacked are skipped.

Result: <dest>/ANPHY-Sleep/osfstorage/{EPCTL01/..., EPCTL02/..., *.pos, ...}

examples:
  python scripts/download_anphy.py --list
  python scripts/download_anphy.py
  python scripts/download_anphy.py --subjects EPCTL01 EPCTL02
  python scripts/download_anphy.py --dest /big/disk/anphy     then: ln -s /big/disk/anphy data/anphy
"""


def get_json(url: str) -> dict:
    """Fetch and decode one JSON document.

    Parameters
    ----------
    url : str
        Address to fetch.

    Returns
    -------
    dict
        The decoded response.
    """
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_files() -> list[dict]:
    """List every file at the top of the OSF storage, following pagination.

    Returns
    -------
    list of dict
        ``name``, ``size`` (bytes), ``md5`` and download ``url`` of each file.
    """
    files, url = [], LISTING
    while url:
        page = get_json(url)
        for entry in page["data"]:
            attributes = entry["attributes"]
            if attributes["kind"] == "file":
                files.append({"name": attributes["name"], "size": attributes["size"],
                              "md5": attributes["extra"]["hashes"].get("md5"),
                              "url": entry["links"]["download"]})
        url = page["links"].get("next")
    return files


def subject_of(name: str) -> str | None:
    """Subject id of a release file.

    Parameters
    ----------
    name : str
        File name, such as ``EPCTL01-2025.zip``.

    Returns
    -------
    str or None
        ``EPCTL01``, or ``None`` for files that are not a subject archive.
    """
    return name[:7] if name.startswith("EPCTL") and name.endswith(".zip") else None


def md5sum(path: Path) -> str:
    """MD5 digest of a file, read in chunks.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict, target: Path, retries: int = 10) -> None:
    """Fetch one file, resuming a partial ``.part`` file with an HTTP range request.

    Parameters
    ----------
    entry : dict
        File description from :func:`list_files`.
    target : Path
        Final location; the file is renamed there only after the checks pass.
    retries : int
        Attempts before giving up.

    Raises
    ------
    RuntimeError
        If the file still fails the size or md5 check after ``retries``.
    """
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have < entry["size"]:
            request = urllib.request.Request(entry["url"], headers={"Range": f"bytes={have}-"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if have and response.status != 206:
                        have = 0
                    with partial.open("ab" if have else "wb") as handle:
                        done, last = have, 0.0
                        while block := response.read(CHUNK):
                            handle.write(block)
                            done += len(block)
                            if time.time() - last > 2:
                                last = time.time()
                                print(f"\r  {entry['name']}: {done / 1e9:6.2f} / "
                                      f"{entry['size'] / 1e9:.2f} GB", end="", flush=True)
                print()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                print(f"\n  interrupted ({error}); retry {attempt}/{retries}")
                time.sleep(min(60, 5 * attempt))
                continue

        if partial.stat().st_size == entry["size"] and (
                not entry["md5"] or md5sum(partial) == entry["md5"]):
            partial.rename(target)
            return
        print(f"  {entry['name']}: size or md5 mismatch, downloading again")
        partial.unlink()
    raise RuntimeError(f"could not download {entry['name']}")


def unpack(archive: Path, root: Path) -> None:
    """Extract a zip into ``root``; its members already carry their folder name.

    Parameters
    ----------
    archive : Path
        Zip file.
    root : Path
        Destination folder.
    """
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)


def is_unpacked(root: Path, entry: dict) -> bool:
    """Whether the content of a release file is already on disk.

    Plain files are checked for existence only, because local copies may have
    been edited (one copy of the montage lost its electrode-count header).

    Parameters
    ----------
    root : Path
        The ``osfstorage`` folder.
    entry : dict
        File description from :func:`list_files`.

    Returns
    -------
    bool
        True if a subject folder holds an EDF and a hypnogram, an archive's
        folder exists, or a plain file exists.
    """
    name, subject = entry["name"], subject_of(entry["name"])
    if subject:
        folder = root / subject
        return folder.is_dir() and any(folder.glob("*.edf")) and any(folder.glob("*.txt"))
    if name.endswith(".zip"):
        return (root / name[:-4]).is_dir()
    return (root / name).exists()


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("data/anphy"),
                        help="folder that will hold ANPHY-Sleep/osfstorage (default: data/anphy)")
    parser.add_argument("--subjects", nargs="*",
                        help="only these subjects, e.g. EPCTL01 EPCTL02 (default: all)")
    parser.add_argument("--no-subjects", action="store_true",
                        help="only the small shared files (montage, readme, tables, code)")
    parser.add_argument("--keep-zips", action="store_true", help="keep archives after unpacking")
    parser.add_argument("--list", action="store_true", help="list the release and exit")
    return parser.parse_args()


def main() -> None:
    """Download and unpack the selected part of the release."""
    args = parse_args()
    files = list_files()
    if args.list:
        for entry in files:
            print(f"{entry['size'] / 1e9:8.3f} GB  {entry['name']}")
        print(f"{sum(e['size'] for e in files) / 1e9:8.1f} GB  total (compressed)")
        return

    wanted = {s.upper() for s in args.subjects} if args.subjects else None
    selected = []
    for entry in files:
        subject = subject_of(entry["name"])
        if subject and (args.no_subjects or (wanted is not None and subject not in wanted)):
            continue
        selected.append(entry)
    if wanted:
        missing = wanted - {subject_of(e["name"]) for e in selected}
        if missing:
            sys.exit(f"not in the release: {sorted(missing)}")

    root = args.dest / "ANPHY-Sleep" / "osfstorage"
    root.mkdir(parents=True, exist_ok=True)
    todo = [e for e in selected if not is_unpacked(root, e)]
    need = sum(e["size"] for e in todo)
    free = shutil.disk_usage(root).free
    print(f"{len(selected) - len(todo)} already present, {len(todo)} to fetch "
          f"({need / 1e9:.1f} GB compressed; {free / 1e9:.0f} GB free in {root})")
    if todo and free < 2.6 * need:
        print("warning: unpacked subjects take about 1.5x their zip size, and each zip is kept "
              "until it is unpacked")

    for index, entry in enumerate(todo, start=1):
        print(f"[{index}/{len(todo)}] {entry['name']}")
        target = root / entry["name"]
        if not target.exists():
            download(entry, target)
        if entry["name"].endswith(".zip"):
            print(f"  unpacking {entry['name']}")
            unpack(target, root)
            if not args.keep_zips:
                target.unlink()
    print(f"done: {root}")


if __name__ == "__main__":
    main()
