import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import fi_config as C
import fi_core as K
import fi_select as S
from s20_precision_chance_cnn_test import build, SETS

DETECTORS = ["D0_FI", "D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest", "D5_LR_hyst"]
LEVELS = [("any", 0.0), ("25", 0.25), ("50", 0.50)]
LONG_S = 4.0
N_BOOT = 1000


def shift_offsets(pool, n_shifts, seed):
    rng = np.random.RandomState(seed)
    starts = np.where(pool.rec_first)[0]
    ends = np.concatenate([starts[1:], [len(pool.rec)]])
    lens = ends - starts
    off = np.zeros((n_shifts, len(starts)), dtype=np.int64)
    for i in range(n_shifts):
        for j, L in enumerate(lens):
            off[i, j] = rng.randint(1, L) if L > 1 else 0
    return starts, lens, off


def shifted(cue, starts, lens, k):
    rid = np.repeat(np.arange(len(starts)), lens)
    pos = np.arange(len(cue)) - starts[rid]
    return cue[starts[rid] + (pos - k[rid]) % lens[rid]]


def fractions(pool, cue):
    cs = np.concatenate([[0], np.cumsum(cue.astype(np.int64))])
    n = cs[pool.ep_end + 1] - cs[pool.ep_start]
    return n / (pool.ep_end - pool.ep_start + 1.0)


def hit(frac, level):
    return frac > 0 if level == 0.0 else frac >= level


def pooled(subj, h, keep):
    return 100.0 * h[keep].mean() if keep.any() else np.nan


def boot_ci(subj, h, rng):
    subs = np.unique(subj)
    n_ep = np.array([(subj == s).sum() for s in subs], dtype=float)
    n_hit = np.array([h[subj == s].sum() for s in subs], dtype=float)
    out = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(subs), len(subs))
        if n_ep[i].sum():
            out.append(100.0 * n_hit[i].sum() / n_ep[i].sum())
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def patient_level(subj, h_obs, h_ch):
    rows = []
    for s in np.unique(subj):
        m = subj == s
        o = 100.0 * h_obs[m].mean()
        c = 100.0 * h_ch[:, m].mean(axis=1)
        rows.append(dict(subject=s, episodes=int(m.sum()), observed=o, chance=c.mean(),
                         p_patient=(1 + (c >= o).sum()) / (len(c) + 1.0)))
    return pd.DataFrame(rows)


def wilcoxon_p(d):
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    try:
        return wilcoxon(d, alternative="greater").pvalue
    except TypeError:
        p = wilcoxon(d).pvalue
        return p / 2 if d.mean() > 0 else 1 - p / 2


def analyse(ds, name, pool, cue, rng, rows, prow):
    base = K.evaluate(pool, cue)
    starts, lens, off = shift_offsets(pool, C.N_SHIFTS, C.SEED)
    obs = fractions(pool, cue)
    ch = np.empty((C.N_SHIFTS, len(obs)), dtype=np.float32)
    for i in range(C.N_SHIFTS):
        ch[i] = fractions(pool, shifted(cue, starts, lens, off[i]))
    dur = (pool.ep_end - pool.ep_start + 1) * C.hop_s()
    subj = pool.subject[pool.ep_start]
    for stratum, keep in [("all", np.ones(len(obs), bool)), ("long", dur >= LONG_S)]:
        if not keep.any():
            continue
        for lab, lv in LEVELS:
            h_obs = hit(obs, lv)
            h_ch = hit(ch, lv)
            o = pooled(subj, h_obs, keep)
            c = 100.0 * h_ch[:, keep].mean(axis=1)
            lo, hi = boot_ci(subj[keep], h_obs[keep], rng)
            pt = patient_level(subj[keep], h_obs[keep], h_ch[:, keep])
            d = (pt["observed"] - pt["chance"]).values
            rows.append(dict(dataset=ds, detector=name, stratum=stratum, level=lab,
                             episodes=int(keep.sum()), patients=len(pt), observed_pct=o,
                             ci_lo=lo, ci_hi=hi, chance_pct=c.mean(), gain_points=o - c.mean(),
                             p_one_sided=(1 + (c >= o).sum()) / (C.N_SHIFTS + 1.0),
                             patients_above_chance=int((d > 0).sum()),
                             patients_p05=int((pt["p_patient"] < 0.05).sum()),
                             median_patient_gain=float(np.median(d)), wilcoxon_p=wilcoxon_p(d),
                             fa_per_h=base["fa_per_h"], false_on_pct=base["false_on_%"]))
            pt.insert(0, "level", lab)
            pt.insert(0, "stratum", stratum)
            pt.insert(0, "detector", name)
            pt.insert(0, "dataset", ds)
            prow.append(pt)
        o = 100.0 * obs[keep].mean()
        c = 100.0 * ch[:, keep].mean(axis=1)
        rows.append(dict(dataset=ds, detector=name, stratum=stratum, level="mean_fraction",
                         episodes=int(keep.sum()), patients=len(np.unique(subj[keep])),
                         observed_pct=o, chance_pct=c.mean(), gain_points=o - c.mean(),
                         p_one_sided=(1 + (c >= o).sum()) / (C.N_SHIFTS + 1.0),
                         fa_per_h=base["fa_per_h"], false_on_pct=base["false_on_%"]))
    return base["coverage_%"]


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    v1 = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    v2p = os.path.join(C.OUT, "v2", "frozen_v2.json")
    use_v2 = os.path.exists(v2p)
    if use_v2:
        import fi2_features as F2
        from s11_v2_develop import score as v2_score
        v2 = json.load(open(v2p))
        model = pickle.load(open(os.path.join(C.OUT, "v2", v2["model"]["file"]), "rb"))
    rng = np.random.RandomState(C.SEED)
    rows, prow = [], []
    for folder, ds, ctx in SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            print("missing", rp)
            continue
        recs = pd.read_csv(rp)
        g = recs[recs["dataset"] == ds]
        if len(g) == 0:
            continue
        pool, _, _ = build(folder, g)
        lrp = json.load(open(os.path.join(C.OUT, "frozen", v1[ctx]["lr_params"]["file"])))
        s_lr = (((pool.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"]) + lrp["intercept"])
        for name in DETECTORS:
            s = s_lr if name.startswith("D5") else pool.F[:, 0]
            cue = S.apply(pool, s, pool.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET])
            cov = analyse(ds, name, pool, cue, rng, rows, prow)
            print(ds, name, "coverage check %.1f" % cov, flush=True)
        if use_v2:
            X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
            p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
            cue = S.apply(p2, v2_score(model, X), None, v2[ctx]["%g" % C.MAIN_BUDGET])
            cov = analyse(ds, "V2_GBT_hyst", p2, cue, rng, rows, prow)
            print(ds, "V2_GBT_hyst", "coverage check %.1f" % cov, flush=True)
    a = pd.DataFrame(rows)
    b = pd.concat(prow, ignore_index=True) if prow else pd.DataFrame()
    a.to_csv(os.path.join(out, "strict_coverage.csv"), index=False)
    b.to_csv(os.path.join(out, "strict_coverage_patients.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    cols = ["dataset", "detector", "stratum", "level", "episodes", "patients", "observed_pct", "ci_lo", "ci_hi",
            "chance_pct", "gain_points", "p_one_sided", "patients_above_chance", "patients_p05",
            "median_patient_gain", "wilcoxon_p"]
    print("\nStrict episode coverage (post hoc)")
    print(a[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
