"""Python reference of the frozen D3/D4 controller for one recording (uses fi_core / fi_select)."""
import struct
import numpy as np


def params_bytes(cfg, arrest):
    a = cfg.get("arrest") if arrest else None
    g = cfg["g"]
    return struct.pack("<3f3i2fi", cfg["on"], cfg["off"], g, int(cfg["k"]), int(cfg["hold"]),
                       1 if a else 0, a[0] if a else 0.0, a[1] if a else 0.0, int(a[2]) if a else 1)


def ref_cue(K, acc, cfg, arrest):
    F, idx = K.hop_features(acc)
    n = len(F)
    rec_first = np.zeros(n, bool); rec_first[0] = True
    rec_start = np.zeros(n, int); rec_end = np.full(n, n - 1)
    x = K.moving_average(F[:, 0], rec_start, cfg["k"])
    x = np.where(F[:, 1] > cfg["g"], x, -np.inf)
    cue = K.hysteresis(x, cfg["on"], cfg["off"], rec_first)
    cue = K.min_hold(cue, rec_end, cfg["hold"], rec_first)
    if arrest:
        drop, walk, hold = cfg["arrest"]
        a = F[:, K.FEATURES.index("arrest")]
        past_level = F[:, 1] - a
        raw = (-a > drop) & (past_level > walk)
        cue = cue | K.min_hold(raw, rec_end, hold, rec_first)
    return cue, F, x
