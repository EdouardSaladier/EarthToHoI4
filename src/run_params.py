"""Generation parameters: everything a run depends on, in one JSON file, so a
user's crash can be reproduced on another machine.

The file holds the full request (selected countries, frame, every tab's
settings, the mod settings), the server-side settings (Land/Sea tab, country
colours), the web UI's saved state, and what the run ran on (tool code
version, Python and packages, OS, game version, datasets), plus the result of
the run (error and traceback if it failed) and the last server errors.

- Every HOI4 export writes it before starting: into runs/ (the last
  KEEP_RUNS kept) and into the exported mod's folder, then completes it with
  the result.
- The web UI's "Save parameters" / "Load parameters" buttons download the
  current parameters / restore them from such a file.
- Replay (for debugging):
      python src/run_params.py replay <file.json> [--export "Mod name"]
  applies the file's settings (backing up and restoring the local ones),
  runs every tab for its request and, with --export, writes the mod.
"""
import collections
import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
import threading
import traceback
from pathlib import Path

CODE = Path(__file__).resolve().parent   # src/
ROOT = CODE.parent                       # the project (data/, runs/, .git)
RUNS_DIR = ROOT / "runs"
KEEP_RUNS = 30
FORMAT = "earthtohoi4-params"
VERSION = 1
PACKAGES = ("flask", "numpy", "scipy", "pillow", "shapely", "pyshp", "opencv-python", "unidecode")

# the last server errors (tracebacks), filled by app.py
RECENT_ERRORS = collections.deque(maxlen=10)
_lock = threading.Lock()


def record_error(where, exc):
    RECENT_ERRORS.append({"time": _now(), "where": where, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))})


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def code_version():
    """A content hash of the tool's code (*.py + templates), and the git commit
    if the folder is a clone (with a flag for local changes)."""
    h = hashlib.sha256()
    for path in sorted(list(CODE.glob("*.py")) + list((CODE / "templates").glob("*.html"))):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    out = {"code_hash": h.hexdigest()[:16]}
    import shutil
    git = shutil.which("git") or next((str(p) for p in (Path(r"C:\Program Files\Git\cmd\git.exe"),
                                                         Path(r"C:\Program Files (x86)\Git\cmd\git.exe"))
                                       if p.exists()), None)
    if git and (ROOT / ".git").exists():
        try:
            commit = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
            if commit.returncode == 0:
                out["git_commit"] = commit.stdout.strip()
                dirty = subprocess.run([git, "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                       capture_output=True, text=True, timeout=5)
                out["git_local_changes"] = bool(dirty.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def environment():
    import importlib.metadata as md
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    return {"python": sys.version.split()[0], "os": platform.platform(), "packages": pk}


def datasets():
    """Name, size and date of every dataset file (what a run read)."""
    out = {}
    data = ROOT / "data"
    if data.is_dir():
        for d in sorted(p for p in data.iterdir() if p.is_dir()):
            files = [f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")]
            out[d.name] = {f.name: [f.stat().st_size, _dt.datetime.fromtimestamp(f.stat().st_mtime).date().isoformat()]
                           for f in sorted(files)}
    return out


def collect(query, ui=None, kind="snapshot"):
    """The parameters of a run for `query` (the full request's query string)."""
    import core
    import hoi4mod
    import palettes
    game = hoi4mod.find_game_dir()
    return {"format": FORMAT, "version": VERSION, "kind": kind, "created": _now(),
            "query": query, "ui": ui or {},
            "settings": {"land_sea": core.load_land_sea_settings(), "colors": palettes.load_config(),
                         "game_dir": hoi4mod.PATHS.get("game"), "mod_dir": hoi4mod.PATHS.get("mod")},
            "tool": code_version(), "environment": environment(),
            "game": {"found": bool(game), "version": hoi4mod.game_version(game) if game else None},
            "datasets": datasets(),
            "recent_errors": list(RECENT_ERRORS)}


def write(params, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def save_run(params, label="run"):
    """Writes the parameters into runs/ (oldest beyond KEEP_RUNS removed)."""
    with _lock:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40] or "run"
        path = write(params, RUNS_DIR / f"{stamp}_{safe}.json")
        runs = sorted(RUNS_DIR.glob("*.json"))
        for old in runs[:-KEEP_RUNS]:
            try:
                old.unlink()
            except OSError:
                pass
        return path


def finish(params, paths, state, error=None, tb=None, **extra):
    """Completes a run's parameters with its result, in every file written."""
    params["result"] = {"state": state, "finished": _now(), **extra}
    if error:
        params["result"]["error"] = error
        params["result"]["traceback"] = tb
    params["recent_errors"] = list(RECENT_ERRORS)
    for p in paths:
        try:
            write(params, p)
        except OSError:
            pass


def check(params):
    """Validates a loaded file; returns a list of notes (differences with this
    installation worth knowing when reproducing)."""
    if not isinstance(params, dict) or params.get("format") != FORMAT:
        raise ValueError("Not an EarthToHoI4 parameters file")
    if int(params.get("version", 0)) > VERSION:
        raise ValueError(f"Parameters file version {params.get('version')} is newer than this tool's ({VERSION})")
    notes = []
    here = code_version()
    if params.get("tool", {}).get("code_hash") not in (None, here["code_hash"]):
        notes.append("made with another version of the tool's code"
                     + (f" (commit {params['tool'].get('git_commit', '?')[:10]})" if params["tool"].get("git_commit") else ""))
    theirs = params.get("datasets") or {}
    ours = datasets()
    diff = sorted(k for k in set(theirs) | set(ours) if theirs.get(k) != ours.get(k))
    if diff:
        notes.append("datasets differ: " + ", ".join(diff))
    return notes


def apply_settings(params):
    """The file's Land/Sea settings and colours become the active ones."""
    import core
    import palettes
    s = params.get("settings") or {}
    if s.get("land_sea") is not None:
        core.save_land_sea_settings(s["land_sea"])
    if s.get("colors") is not None:
        palettes.save_config({"preset": s["colors"].get("preset"), "overrides": s["colors"].get("overrides", {})})


def _replay(path, export_name=None):
    import core
    import palettes
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    for n in check(params):
        print("note:", n)
    backup = {p: (p.read_bytes() if p.exists() else None)
              for p in (core.LAND_SEA_SETTINGS_PATH, palettes.CONFIG_PATH)}
    try:
        apply_settings(params)
        import app
        import hoi4mod
        with app.app.test_request_context("/?" + params["query"]):
            src = app._hoi4_sources()
        print("every tab ran")
        if export_name:
            out = hoi4mod.mod_dir() / hoi4mod.ascii_name(export_name)
            from urllib.parse import parse_qs
            q = parse_qs(params["query"])
            settings = hoi4mod.normalize_settings(dict(
                {k: q["mod_" + k][0] for k in hoi4mod.MOD_DEFAULTS if "mod_" + k in q}, name=export_name))
            rep = hoi4mod.build_mod(src, settings, out, hoi4mod.find_game_dir(), progress=lambda m: print("  ", m))
            hoi4mod.install_descriptor(out, hoi4mod.mod_dir())
            print("exported to", out, "problems:", rep.get("problems"))
    finally:
        for p, data in backup.items():
            if data is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(data)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "replay":
        name = None
        if "--export" in sys.argv:
            name = sys.argv[sys.argv.index("--export") + 1]
        _replay(sys.argv[2], name)
    else:
        print(__doc__)
