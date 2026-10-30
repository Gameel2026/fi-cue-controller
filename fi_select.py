import itertools
import numpy as np
import fi_config as C
import fi_core as K


def quantile_grid(s, mask, lo, hi, n):
    v = s[mask & np.isfinite(s)]
    return np.unique(np.round(np.percentile(v, np.linspace(lo, hi, n)), 4))


def run_grid(pool, s, gate, configs):
    cache = {}
    rows = []
    for cfg in configs:
        key = (cfg["k"], cfg.get("g"))
        if key not in cache:
            x = K.moving_average(s, pool.rec_start, cfg["k"])
            if cfg.get("g") is not None:
                x = np.where(gate > cfg["g"], x, -np.inf)
            cache[key] = x
        x = cache[key]
        cue = K.hysteresis(x, cfg["on"], cfg["off"], pool.rec_first)
        cue = K.min_hold(cue, pool.rec_end, cfg["hold"], pool.rec_first)
        if cfg.get("arrest") is not None:
            cue = cue | arrest_cue(pool, cfg["arrest"])
        r = K.evaluate(pool, cue)
        rows.append(r)
    return rows


def arrest_cue(pool, a):
    drop, walk, hold = a
    arrest = pool.F[:, K.FEATURES.index("arrest")]
    past_level = pool.F[:, K.FEATURES.index("logP4")] - arrest
    raw = (-arrest > drop) & (past_level > walk)
    return K.min_hold(raw, pool.rec_end, hold, pool.rec_first)


def choose(rows, configs, budget):
    fa = np.array([r["fa_per_h"] for r in rows])
    cov = np.array([r["coverage_%"] for r in rows])
    tim = np.array([r["timely_%"] for r in rows])
    fo = np.array([r["false_on_%"] for r in rows])
    ok = (fa <= budget) & (fo <= 100.0 * (1.0 - C.SPEC_FLOOR))
    if ok.any():
        idx = np.where(ok)[0]
        order = np.lexsort((fa[idx], -tim[idx], -cov[idx]))
        return configs[idx[order[0]]]
    return configs[int(np.argmin(fa + 1000.0 * (fo > 100.0 * (1.0 - C.SPEC_FLOOR))))]


def apply(pool, s, gate, cfg):
    x = K.moving_average(s, pool.rec_start, cfg["k"])
    if cfg.get("g") is not None:
        x = np.where(gate > cfg["g"], x, -np.inf)
    cue = K.hysteresis(x, cfg["on"], cfg["off"], pool.rec_first)
    cue = K.min_hold(cue, pool.rec_end, cfg["hold"], pool.rec_first)
    if cfg.get("arrest") is not None:
        cue = cue | arrest_cue(pool, cfg["arrest"])
    return cue


def configs_for(level, s, gate, mask, base=None):
    th = quantile_grid(s, mask, 50, 99.7, 30)
    gates = [None] if gate is None else [float(v) for v in quantile_grid(gate, mask, 10, 50, 4)]
    out = []
    if level == "plain":
        for t in th:
            out.append({"on": float(t), "off": float(t), "k": 1, "hold": 1, "g": None})
    elif level == "gate":
        for t, g in itertools.product(th, gates):
            out.append({"on": float(t), "off": float(t), "k": 1, "hold": 1, "g": g})
    elif level == "smooth":
        for t, g, k in itertools.product(th, gates, [4, 8]):
            out.append({"on": float(t), "off": float(t), "k": k, "hold": 1, "g": g})
    elif level == "hyst":
        spread = np.nanstd(s[mask & np.isfinite(s)])
        for t, g, k, d, h in itertools.product(th, gates, [1, 4, 8], [0.25, 0.5, 1.0], [4, 12]):
            out.append({"on": float(t), "off": float(t - d * spread), "k": k, "hold": h, "g": g})
    elif level == "arrest":
        walk_levels = quantile_grid(gate, mask, 50, 80, 3)
        for d, w, h in itertools.product([1.0, 1.5, 2.0, 3.0], walk_levels, [8, 16]):
            c = dict(base)
            c["arrest"] = (float(d), float(w), int(h))
            out.append(c)
    return out
