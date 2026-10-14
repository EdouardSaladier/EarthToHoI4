"""Turns pixel boundaries / pixel lines into smoothed polylines."""
import numpy as np


def _walk_chains(a, b):
    """Chain an undirected edge list (vertex ids a[i]-b[i]) into vertex sequences.

    Chains break at vertices whose degree != 2. Pure cycles are returned closed
    (first vertex repeated at the end).
    """
    if len(a) == 0:
        return []
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    m = len(a)
    ends = np.concatenate([a, b])
    eidx = np.concatenate([np.arange(m), np.arange(m)])
    order = np.argsort(ends, kind="stable")
    sv = ends[order]
    se = eidx[order]
    uniq, start, counts = np.unique(sv, return_index=True, return_counts=True)
    chains = []
    # python-level adjacency for speed of walking
    start_l = start.tolist()
    count_l = counts.tolist()
    se_l = se.tolist()
    a_l = a.tolist()
    b_l = b.tolist()
    vpos = {v: i for i, v in enumerate(uniq.tolist())}
    used_l = [False] * m

    def other(e, v):
        return b_l[e] if a_l[e] == v else a_l[e]

    def walk(v0, e0):
        seq = [v0]
        v = v0
        e = e0
        while True:
            used_l[e] = True
            v = other(e, v)
            seq.append(v)
            k = vpos[v]
            if count_l[k] != 2:
                break
            s = start_l[k]
            e1, e2 = se_l[s], se_l[s + 1]
            nxt = e2 if e1 == e else e1
            if used_l[nxt]:
                break
            e = nxt
        return seq

    # open chains starting at junctions / endpoints
    for v, k in vpos.items():
        if count_l[k] != 2:
            s = start_l[k]
            for j in range(count_l[k]):
                e = se_l[s + j]
                if not used_l[e]:
                    chains.append(walk(v, e))
    # remaining cycles
    for e in range(m):
        if not used_l[e]:
            chains.append(walk(a_l[e], e))
    return chains


def chaikin(pts, iterations=2, closed=False):
    """Chaikin corner cutting. pts: (n, k) array (extra columns are interpolated too)."""
    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            return pts
        p0 = pts[:-1]
        p1 = pts[1:]
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        mid = np.empty((2 * (n - 1), pts.shape[1]), dtype=pts.dtype)
        mid[0::2] = q
        mid[1::2] = r
        if closed:
            pts = np.vstack([mid, mid[:1]])
        else:
            pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def label_boundaries(labels, mask_fn=None):
    """Boundary edges between differing labels, on the pixel-corner grid.

    mask_fn(la, lb) -> bool array selecting which differing pairs to keep.
    Returns list of (n, 2) float arrays of corner coordinates (x, y), in map pixels.
    """
    H, W = labels.shape
    W1 = W + 1
    # vertical edges between (y, x-1) and (y, x): from corner (x, y) to (x, y+1)
    la = labels[:, :-1]
    lb = labels[:, 1:]
    sel = la != lb
    if mask_fn is not None:
        sel &= mask_fn(la, lb)
    ys, xs = np.nonzero(sel)
    xs = xs + 1
    va = ys.astype(np.int64) * W1 + xs
    vb = (ys.astype(np.int64) + 1) * W1 + xs
    # horizontal edges between (y-1, x) and (y, x): from corner (x, y) to (x+1, y)
    la = labels[:-1, :]
    lb = labels[1:, :]
    sel = la != lb
    if mask_fn is not None:
        sel &= mask_fn(la, lb)
    ys, xs = np.nonzero(sel)
    ys = ys + 1
    ha = ys.astype(np.int64) * W1 + xs
    hb = ys.astype(np.int64) * W1 + xs + 1
    a = np.concatenate([va, ha])
    b = np.concatenate([vb, hb])
    chains = _walk_chains(a, b)
    out = []
    for c in chains:
        c = np.asarray(c, dtype=np.int64)
        pts = np.stack([(c % W1).astype(np.float64), (c // W1).astype(np.float64)], axis=1)
        out.append(pts)
    return out


def smooth_boundary(pts, iterations=3):
    """Smooth a staircase corner chain: use edge midpoints, keep chain ends."""
    closed = len(pts) > 3 and np.all(pts[0] == pts[-1])
    if len(pts) < 3:
        return pts
    mids = 0.5 * (pts[:-1] + pts[1:])
    if closed:
        mids = np.vstack([mids, mids[:1]])
        return chaikin(mids, iterations, closed=True)
    mids = np.vstack([pts[:1], mids, pts[-1:]])
    return chaikin(mids, iterations, closed=False)


def pixel_lines(mask):
    """Chains of 'on' pixels (e.g. rivers). Returns list of (y, x) int index arrays."""
    H, W = mask.shape
    if not mask.any():
        return []
    a_list, b_list = [], []
    m = mask
    # orthogonal neighbours
    right = np.zeros_like(m)
    right[:, :-1] = m[:, :-1] & m[:, 1:]
    yy, xx = np.nonzero(right)
    a_list.append(yy.astype(np.int64) * W + xx)
    b_list.append(yy.astype(np.int64) * W + xx + 1)
    down = np.zeros_like(m)
    down[:-1, :] = m[:-1, :] & m[1:, :]
    yy, xx = np.nonzero(down)
    a_list.append(yy.astype(np.int64) * W + xx)
    b_list.append((yy.astype(np.int64) + 1) * W + xx)
    # diagonals only when not already connected through an orthogonal pixel
    dr = np.zeros_like(m)
    dr[:-1, :-1] = m[:-1, :-1] & m[1:, 1:] & ~m[:-1, 1:] & ~m[1:, :-1]
    yy, xx = np.nonzero(dr)
    a_list.append(yy.astype(np.int64) * W + xx)
    b_list.append((yy.astype(np.int64) + 1) * W + xx + 1)
    dl = np.zeros_like(m)
    dl[:-1, 1:] = m[:-1, 1:] & m[1:, :-1] & ~m[:-1, :-1] & ~m[1:, 1:]
    yy, xx = np.nonzero(dl)
    a_list.append(yy.astype(np.int64) * W + xx + 1)
    b_list.append((yy.astype(np.int64) + 1) * W + xx)
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    chains = _walk_chains(a, b)
    out = []
    for c in chains:
        c = np.asarray(c, dtype=np.int64)
        out.append((c // W, c % W))
    return out
