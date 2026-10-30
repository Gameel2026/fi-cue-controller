import numpy as np
import fi_config as C

FEATURES = ["logFI4", "logP4", "logFI4_V", "logFI4_AP", "logFI4_ML", "logFI2", "logP2",
            "arrest", "domfreq", "stepreg", "meanV", "meanAP"]
EPS = 1e-8


def frames(x, win, hop):
    pad = np.repeat(x[:1], win - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    n_hops = (len(x) - 1) // hop + 1
    ends = np.arange(n_hops) * hop + (win - 1)
    idx = ends[:, None] - np.arange(win - 1, -1, -1)[None, :]
    return xp[idx], ends - (win - 1)


def band_powers(fr, fs):
    win = fr.shape[1]
    w = np.hanning(win)[None, :, None]
    xd = (fr - fr.mean(axis=1, keepdims=True)) * w
    P = np.abs(np.fft.rfft(xd, axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1.0 / fs)
    loc = (f >= C.LOC_BAND[0]) & (f < C.LOC_BAND[1])
    frz = (f >= C.FRZ_BAND[0]) & (f <= C.FRZ_BAND[1])
    tot = (f >= C.LOC_BAND[0]) & (f <= C.FRZ_BAND[1])
    return P[:, loc, :].sum(axis=1), P[:, frz, :].sum(axis=1), P[:, tot, :].sum(axis=1), P, f


def hop_features(acc):
    fs = C.FS
    fr4, sample_idx = frames(acc, C.WIN_LONG, C.HOP)
    loc4, frz4, tot4, P4, f4 = band_powers(fr4, fs)
    fr2, _ = frames(acc, C.WIN_SHORT, C.HOP)
    loc2, frz2, tot2, _, _ = band_powers(fr2, fs)
    n = len(fr4)
    F = np.zeros((n, len(FEATURES)), dtype=np.float32)
    F[:, 0] = np.log((frz4.sum(1) + EPS) / (loc4.sum(1) + EPS))
    F[:, 1] = np.log(tot4.sum(1) + EPS)
    F[:, 2] = np.log((frz4[:, 0] + EPS) / (loc4[:, 0] + EPS))
    F[:, 3] = np.log((frz4[:, 2] + EPS) / (loc4[:, 2] + EPS))
    F[:, 4] = np.log((frz4[:, 1] + EPS) / (loc4[:, 1] + EPS))
    F[:, 5] = np.log((frz2.sum(1) + EPS) / (loc2.sum(1) + EPS))
    F[:, 6] = np.log(tot2.sum(1) + EPS)
    lp = np.log(loc4.sum(1) + EPS)
    a, b = C.ARREST_PAST
    past = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - a), max(0, i - b)
        if hi > lo:
            past[i] = np.median(lp[lo:hi])
    past[np.isnan(past)] = lp[np.isnan(past)]
    F[:, 7] = lp - past
    band = (f4 >= C.LOC_BAND[0]) & (f4 <= C.FRZ_BAND[1])
    Ps = P4.sum(axis=2)[:, band]
    F[:, 8] = f4[band][np.argmax(Ps, axis=1)]
    v = fr4[:, :, 0] - fr4[:, :, 0].mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(v, n=2 * C.WIN_LONG, axis=1)) ** 2
    ac = np.fft.irfft(spec, axis=1)[:, :C.WIN_LONG]
    ac = ac / (ac[:, :1] + EPS)
    l0, l1 = int(0.4 * fs), int(1.4 * fs)
    F[:, 9] = ac[:, l0:l1].max(axis=1)
    F[:, 10] = fr4[:, -C.HOP:, 0].mean(axis=1)
    F[:, 11] = fr4[:, -C.HOP:, 2].mean(axis=1)
    return F, sample_idx


def moving_average(s, rec_start, k):
    if k <= 1:
        return s
    cs = np.concatenate([[0.0], np.cumsum(s, dtype=np.float64)])
    i = np.arange(len(s))
    lo = np.maximum(i - k + 1, rec_start)
    return ((cs[i + 1] - cs[lo]) / (i + 1 - lo)).astype(np.float32)


def hysteresis(s, on, off, rec_first):
    st = np.full(len(s), -1, dtype=np.int8)
    st[s > on] = 1
    st[s < off] = 0
    st[rec_first & (st == -1)] = 0
    idx = np.where(st >= 0, np.arange(len(s)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = st[idx]
    out[out < 0] = 0
    return out.astype(bool)


def run_bounds(x, rec_first):
    x = x.astype(bool)
    prev = np.concatenate([[False], x[:-1]])
    prev[rec_first] = False
    starts = np.where(x & ~prev)[0]
    nxt = np.concatenate([x[1:], [False]])
    last_of_rec = np.concatenate([rec_first[1:], [True]])
    nxt[last_of_rec] = False
    ends = np.where(x & ~nxt)[0]
    return starts, ends


def min_hold(cue, rec_end, hold, rec_first):
    if hold <= 1:
        return cue
    starts, _ = run_bounds(cue, rec_first)
    d = np.zeros(len(cue) + 1, dtype=np.int32)
    stop = np.minimum(starts + hold, rec_end[starts] + 1)
    np.add.at(d, starts, 1)
    np.add.at(d, stop, -1)
    return cue | (np.cumsum(d)[:-1] > 0)


class Pool:
    def __init__(self, F, fog, score, ev, task, rec, subject, keep=None):
        self.F, self.fog, self.score, self.ev, self.task = F, fog, score, ev, task
        self.rec, self.subject = rec, subject
        n = len(rec)
        self.rec_first = np.concatenate([[True], rec[1:] != rec[:-1]])
        starts = np.where(self.rec_first)[0]
        ends = np.concatenate([starts[1:] - 1, [n - 1]])
        rid = np.cumsum(self.rec_first) - 1
        self.rec_start = starts[rid]
        self.rec_end = ends[rid]
        ep = (fog > 0) & score
        self.ep_start, self.ep_end = run_bounds(ep, self.rec_first)
        pre, post = int(round(C.PRE_S / C.hop_s())), int(round(C.POST_S / C.hop_s()))
        d = np.zeros(n + 1, dtype=np.int32)
        a = np.maximum(self.ep_start - pre, self.rec_start[self.ep_start])
        b = np.minimum(self.ep_end + post, self.rec_end[self.ep_end]) + 1
        np.add.at(d, a, 1)
        np.add.at(d, b, -1)
        self.target = np.cumsum(d)[:-1] > 0
        self.pre = pre
        self.hours = score.sum() * C.hop_s() / 3600.0
        self.ep_rec = rec[self.ep_start]

    def subset(self, mask):
        return Pool(self.F[mask], self.fog[mask], self.score[mask], self.ev[mask], self.task[mask],
                    self.rec[mask], self.subject[mask])


def evaluate(pool, cue, detail=False):
    cue = cue.astype(bool)
    cs = np.concatenate([[0], np.cumsum(cue)])
    covered = (cs[pool.ep_end + 1] - cs[pool.ep_start]) > 0
    cstart, cend = run_bounds(cue, pool.rec_first)
    on_at = cue[pool.ep_start]
    j = np.searchsorted(cstart, pool.ep_start, side="right") - 1
    if len(cstart):
        started = np.where(j >= 0, cstart[np.maximum(j, 0)], -10 ** 9)
    else:
        started = np.full(len(j), -10 ** 9)
    timely = on_at & (started >= pool.ep_start - pool.pre)
    tcs = np.concatenate([[0], np.cumsum(pool.target)])
    in_target = (tcs[cend + 1] - tcs[cstart]) > 0
    fa_runs = (~in_target) & pool.score[cstart]
    n_ep = len(pool.ep_start)
    res = {"episodes": n_ep,
           "coverage_%": 100.0 * covered.mean() if n_ep else np.nan,
           "timely_%": 100.0 * timely.mean() if n_ep else np.nan,
           "fa_per_h": fa_runs.sum() / pool.hours if pool.hours > 0 else np.nan,
           "cue_on_%": 100.0 * cue[pool.score].mean(),
           "false_on_%": 100.0 * cue[pool.score & ~pool.target].mean() if (pool.score & ~pool.target).any() else np.nan,
           "hours": pool.hours}
    if detail:
        return res, covered, timely, cstart[fa_runs]
    return res


def controller(pool, s, on, off=None, k=1, hold=1, gate=None, gate_thr=None):
    x = moving_average(s, pool.rec_start, k)
    if gate is not None:
        x = np.where(gate > gate_thr, x, -np.inf)
    cue = hysteresis(x, on, on if off is None else off, pool.rec_first)
    return min_hold(cue, pool.rec_end, hold, pool.rec_first)


def surrogate(pool, cue, n_shifts, seed):
    rng = np.random.RandomState(seed)
    starts = np.where(pool.rec_first)[0]
    ends = np.concatenate([starts[1:], [len(cue)]])
    cov, tim = [], []
    for _ in range(n_shifts):
        sh = np.empty_like(cue)
        for a, b in zip(starts, ends):
            L = b - a
            sh[a:b] = np.roll(cue[a:b], rng.randint(1, L) if L > 1 else 0)
        r = evaluate(pool, sh)
        cov.append(r["coverage_%"])
        tim.append(r["timely_%"])
    return np.array(cov), np.array(tim)
