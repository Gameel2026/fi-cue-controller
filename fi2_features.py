import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K

WINS = [64, 128, 256]
HIGH = (8.0, 16.0)
REL_WIN = 240
LAGS = [4, 8]
EPS = 1e-8


def _bands(fr, fs):
    win = fr.shape[1]
    w = np.hanning(win)[None, :, None]
    xd = (fr - fr.mean(axis=1, keepdims=True)) * w
    P = (np.abs(np.fft.rfft(xd, axis=1)) ** 2).sum(axis=2)
    f = np.fft.rfftfreq(win, 1.0 / fs)
    sel = lambda a, b: P[:, (f >= a) & (f < b)].sum(axis=1)
    return sel(*C.LOC_BAND), sel(C.FRZ_BAND[0], C.FRZ_BAND[1] + 1e-9), sel(*HIGH), P, f


def _running_median(x, rec_start_flags):
    s = pd.Series(x)
    g = np.cumsum(rec_start_flags)
    return s.groupby(g).transform(lambda v: v.rolling(REL_WIN, min_periods=1).median()).to_numpy()


def features(acc):
    fs = C.FS
    cols, names = [], []
    for w in WINS:
        fr, _ = K.frames(acc, w, C.HOP)
        loc, frz, hi, P, f = _bands(fr, fs)
        cols += [np.log(loc + EPS), np.log(frz + EPS), np.log(hi + EPS), np.log((frz + EPS) / (loc + EPS)),
                 np.log((hi + EPS) / (frz + EPS))]
        names += ["lloc%d" % w, "lfrz%d" % w, "lhi%d" % w, "fi%d" % w, "hf%d" % w]
        if w == 256:
            band = (f >= 0.5) & (f < 16)
            Pb = P[:, band] + EPS
            p = Pb / Pb.sum(axis=1, keepdims=True)
            cols += [f[band][np.argmax(Pb, axis=1)], -(p * np.log(p)).sum(axis=1)]
            names += ["domf", "spent"]
            x = fr - fr.mean(axis=1, keepdims=True)
            spec = (np.abs(np.fft.rfft(x, n=2 * w, axis=1)) ** 2).sum(axis=2)
            ac = np.fft.irfft(spec, axis=1)[:, :w]
            ac = ac / (ac[:, :1] + EPS)
            cols.append(ac[:, int(0.4 * fs):int(1.4 * fs)].max(axis=1))
            names.append("stepreg")
    fr, last = K.frames(acc, 64, C.HOP)
    jerk = np.diff(fr, axis=1)
    cols.append(np.log((jerk ** 2).sum(axis=2).mean(axis=1) + EPS))
    names.append("ljerk")
    F = np.stack(cols, axis=1).astype(np.float32)
    return F, names, last


def add_context(F, names, rec):
    first = np.concatenate([[True], rec[1:] != rec[:-1]])
    out, out_names = [F], list(names)
    for nm in ["lloc256", "lfrz256", "lhi256", "lloc128", "ljerk"]:
        j = names.index(nm)
        med = _running_median(F[:, j], first)
        out.append((F[:, j] - med)[:, None])
        out_names.append("rel_" + nm)
    starts = np.where(first)[0]
    rs = starts[np.cumsum(first) - 1]
    idx = np.arange(len(rec))
    for nm in ["fi128", "fi256", "lloc128", "lfrz128", "domf", "stepreg"]:
        j = names.index(nm)
        for L in LAGS:
            prev = np.maximum(idx - L, rs)
            out.append((F[:, j] - F[prev, j])[:, None])
            out_names.append("d%d_%s" % (L, nm))
    keep = [i for i, n in enumerate(out_names) if not (n.startswith(("lloc", "lfrz", "lhi", "ljerk")))]
    X = np.concatenate(out, axis=1)
    return X[:, keep].astype(np.float32), [out_names[i] for i in keep]


def recordings_features(recs, folder):
    Fs, rec_ids, last_all, fog, score, ev, task, subj = [], [], [], [], [], [], [], []
    names = None
    for r in recs.itertuples():
        z = np.load("%s/%s.npz" % (folder, r.rec))
        F, names, last = features(z["acc"].astype(np.float64))
        Fs.append(F)
        rec_ids.append(np.full(len(last), r.rec, dtype=object))
        fog.append(z["fog"][last])
        score.append(z["score"][last])
        ev.append(z["ev"][last])
        task.append(z["task"][last])
        subj.append(np.full(len(last), str(r.subject), dtype=object))
    rec = np.concatenate(rec_ids).astype(str)
    X, xnames = add_context(np.concatenate(Fs), names, rec)
    return X, xnames, rec, np.concatenate(fog), np.concatenate(score).astype(bool), np.concatenate(ev), \
        np.concatenate(task), np.concatenate(subj).astype(str)
