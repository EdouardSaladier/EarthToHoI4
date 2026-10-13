"""Progress of what the tabs are computing, for the bar at the top of the
page (user: "whenever rendering any map, display a progress bar on top, with
an estimated ETA").

Every tab's result goes through `app._stage_get`, so one scope there covers
them all: a stage that has to be computed (a cache miss -- a hit is instant
and shows nothing) opens a scope, and the bar shows its name, how far it is
and how long is left. A request usually runs several stages one after the
other (the Provinces tab needs the render, the rivers, the states, the
terrain ...), numbered as steps.

**The ETA is measured, not guessed**: every finished stage is recorded in
`cache/stage_times.json` as (megapixels, seconds), and a stage's estimate is
the median of its last `SAMPLES` runs, in seconds per megapixel, times this
view's megapixels. Before a stage has ever run here, `BASE_RATE` (measured on
the default view) is used. The bar is time-based, so it moves smoothly
without the renderers having to report anything; a stage that takes longer
than its estimate stops at 99% rather than claiming to be done.
"""
import json
import statistics
import threading
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
TIMES_FILE = CACHE_DIR / "stage_times.json"
SAMPLES = 8                  # runs kept per stage (the median of these is the estimate)
MIN_SECONDS = 0.4            # a stage expected to be faster than this doesn't open the bar
DEFAULT_MPX = 13.0           # a view's size before the first render tells us (the default view)

# Seconds per megapixel of the view, measured on the default view with every
# dataset present. Only used until a stage has run once on this machine.
BASE_RATE = {
    "render": 0.70, "recolor": 0.05, "eez_zones": 0.65, "rivers": 0.25, "cities": 0.15,
    "heightmap": 0.30, "height_levels": 0.15, "density": 0.10, "states": 0.60,
    "terrain": 0.60, "colormap": 0.25, "province_eez": 0.40, "provinces": 6.00, "strategic": 0.30,
}
LABELS = {
    "render": "Country map", "recolor": "Country colours", "eez_zones": "Exclusive economic zones",
    "rivers": "Rivers", "cities": "Cities", "heightmap": "Heightmap", "height_levels": "Heightmap levels",
    "density": "Population density", "states": "States", "terrain": "Terrain", "colormap": "Colour map",
    "province_eez": "Sea zones", "provinces": "Provinces", "strategic": "Strategic regions",
}

_lock = threading.Lock()
_times = None                # {stage: [[megapixels, seconds], ...]}
_view_mpx = DEFAULT_MPX
_job = None                  # the stage running now
_step = 0                    # stages run since the last idle moment
_last_end = 0.0
IDLE_RESET_S = 2.0           # steps start again at 1 after this long with nothing running


def _load():
    global _times
    if _times is None:
        try:
            _times = json.loads(TIMES_FILE.read_text(encoding="utf-8"))
        except Exception:
            _times = {}
    return _times


def _save():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TIMES_FILE.write_text(json.dumps(_times), encoding="utf-8")
    except OSError:
        pass


def set_view_pixels(pixels):
    """The size of the view being worked on: the estimates scale with it."""
    global _view_mpx
    if pixels:
        _view_mpx = max(0.1, pixels / 1e6)


def estimate(stage, mpx=None):
    """Seconds this stage is expected to take on a view of `mpx` megapixels
    (None = the current view), or None when there is nothing to go on."""
    mpx = _view_mpx if mpx is None else mpx
    samples = _load().get(stage) or []
    rates = [s / m for m, s in samples if m > 0 and s > 0]
    rate = statistics.median(rates) if rates else BASE_RATE.get(stage)
    return None if rate is None else max(0.05, rate * mpx)


def record(stage, seconds, mpx=None):
    times = _load()
    times.setdefault(stage, []).append([round(mpx if mpx is not None else _view_mpx, 3), round(seconds, 3)])
    times[stage] = times[stage][-SAMPLES:]
    _save()


class scope:
    """`with render_progress.scope("provinces"):` around a stage's computation."""

    def __init__(self, stage, label=None):
        self.stage = stage
        self.label = label or LABELS.get(stage, stage.replace("_", " ").capitalize())

    def __enter__(self):
        global _job, _step
        with _lock:
            if time.time() - _last_end > IDLE_RESET_S and _job is None:
                _step = 0
            _step += 1
            _job = {"stage": self.stage, "label": self.label, "note": None, "started": time.time(),
                    "estimate": estimate(self.stage), "mpx": _view_mpx, "step": _step}
        return self

    def __exit__(self, *exc):
        global _job, _last_end
        with _lock:
            job = _job
            _job = None
            _last_end = time.time()
        if job is not None and exc[0] is None:
            # the size as it is *now*: the render stage only learns the view's
            # size while it runs (set_view_pixels), so job["mpx"] -- what the
            # estimate had to be made from -- would file it under the last view's
            record(self.stage, time.time() - job["started"])
        return False


def note(message):
    """A line about what the stage is doing now (core's `progress=` callback)."""
    with _lock:
        if _job is not None:
            _job["note"] = str(message).strip()[:120]


def snapshot():
    """What to show in the bar: None when nothing is being computed."""
    with _lock:
        job = dict(_job) if _job else None
    if job is None:
        return {"active": False}
    elapsed = time.time() - job["started"]
    est = job["estimate"]
    if est is None or est < MIN_SECONDS:
        return {"active": True, "label": job["label"], "note": job["note"], "step": job["step"],
                "progress": None, "eta_s": None, "elapsed_s": round(elapsed, 1)}
    return {"active": True, "label": job["label"], "note": job["note"], "step": job["step"],
            "progress": round(min(0.99, elapsed / est), 3),
            "eta_s": max(0, round(est - elapsed)), "elapsed_s": round(elapsed, 1)}
