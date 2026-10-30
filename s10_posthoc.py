import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import fi_config as C
import fi_core as K
import fi_select as S
from s3_ladder import load_pool, fit_lr, lr_score
from s6_freeze import load_context
import s8_external_test as X

N_BOOT = 1000
POSTURE_FREE = ["logFI4", "logP4", "logFI2", "logP2", "arrest", "domfreq", "stepreg"]
MATCH_FALSE_ON = 5.0


def per_subject(pool, cue):
    r, covered, timely, fa_starts = K.evaluate(pool, cue, detail=True)
    subj_ep = pool.subject[pool.ep_start]
    subj_fa = pool.subject[fa_starts]
    subs = np.unique(pool.subject)
    rows = []
    for s in subs:
        m = subj_ep == s
        rows.append([m.sum(), covered[m].sum(), (subj_fa == s).sum(),
                     (pool.score & (pool.subject == s)).sum() * C.hop_s() / 3600.0])
    return np.array(rows, dtype=float), r


def boot(tab, rng):
    n = len(tab)
    cov, fa = [], []
    for _ in range(N_BOOT):
        t = tab[rng.randint(0, n, n)].sum(axis=0)
        cov.append(100 * t[1] / t[0] if t[0] else np.nan)
        fa.append(t[2] / t[3] if t[3] else np.nan)
    return np.nanpercentile(cov, [2.5, 97.5]), np.nanpercentile(fa, [2.5, 97.5])


def boot_diff(ta, tb, rng):
    n = len(ta)
    d = []
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        a, b = ta[i].sum(axis=0), tb[i].sum(axis=0)
        if a[0]:
            d.append(100 * a[1] / a[0] - 100 * b[1] / b[0])
    return np.percentile(d, [2.5, 97.5])


def matched(pool, s):
    best = None
    for on in np.percentile(s[np.isfinite(s)], np.linspace(50, 99.9, 120)):
        cfg = {"on": float(on), "off": float(on - 0.5 * np.nanstd(s)), "k": 4, "hold": 4, "g": None}
        r = K.evaluate(pool, S.apply(pool, s, None, cfg))
        if r["false_on_%"] <= MATCH_FALSE_ON and (best is None or r["coverage_%"] > best["coverage_%"]):
            best = r
    return best


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    rng = np.random.RandomState(C.SEED)
    ci_rows, diff_rows, match_rows, auc_rows = [], [], [], []

    lab = load_context(["tdcsfog", "daphnet"])
    cols = [K.FEATURES.index(f) for f in POSTURE_FREE]
    full = K.Pool(lab.F[:, cols], lab.fog, lab.score, lab.ev, lab.task, lab.rec, lab.subject)
    sc, lr = fit_lr(full, C.SEED)
    s_dev = lr_score(sc, lr, full.F)
    cfgs = S.configs_for("hyst", s_dev, None, full.score)
    pf_cfg = S.choose(S.run_grid(full, s_dev, None, cfgs), cfgs, C.MAIN_BUDGET)

    for ds in C.DEV_DATASETS:
        p = os.path.join(C.OUT, "oof_%s.npz" % ds)
        if not os.path.exists(p):
            continue
        pool = load_pool(ds)
        z = np.load(p)
        cues = {k.split("__")[0]: z[k].astype(bool) for k in z.files if k.endswith("__%g" % C.MAIN_BUDGET)}
        scores = {"FI": pool.F[:, 0], "LR": z["lr"]}
        pc = os.path.join(C.OUT, "oof_cnn_%s.npz" % ds)
        if os.path.exists(pc):
            zc = np.load(pc)
            cues.update({k.split("__")[0]: zc[k].astype(bool) for k in zc.files if k.endswith("__%g" % C.MAIN_BUDGET)})
            scores["CNN"] = zc["cnn"]
        tabs = {}
        for name, cue in cues.items():
            tab, r = per_subject(pool, cue)
            tabs[name] = tab
            ci, fi = boot(tab, rng)
            ci_rows.append({"set": "development", "dataset": ds, "detector": name, "coverage_%": r["coverage_%"],
                            "cov_lo": ci[0], "cov_hi": ci[1], "fa_per_h": r["fa_per_h"], "fa_lo": fi[0], "fa_hi": fi[1],
                            "false_on_%": r["false_on_%"]})
        for a, b in [("D3_FI_gate_smooth_hyst", "D0_FI"), ("D5_LR_hyst", "D3_FI_gate_smooth_hyst"),
                     ("D6_CNN_hyst", "D3_FI_gate_smooth_hyst")]:
            if a in tabs and b in tabs:
                lo, hi = boot_diff(tabs[a], tabs[b], rng)
                diff_rows.append({"set": "development", "dataset": ds, "comparison": "%s - %s" % (a, b), "lo": lo, "hi": hi})
        m = pool.score
        for k, s in scores.items():
            ok = m & np.isfinite(s)
            auc_rows.append({"set": "development", "dataset": ds, "score": k, "auc": roc_auc_score(pool.fog[ok], s[ok])})
            r = matched(pool, s)
            if r:
                match_rows.append({"set": "development", "dataset": ds, "score": k, **r})
        print("development", ds, "done", flush=True)

    with open(X.FROZEN) as fh:
        frozen = json.load(fh)
    with open(os.path.join(C.OUT, "frozen", frozen["lab"]["lr_params"]["file"])) as fh:
        lrp = json.load(fh)
    recs = pd.read_csv(os.path.join(X.EXT, "recordings.csv"))
    for ds, g in recs.groupby("dataset"):
        pool, accs, table = X.build_pool(g)
        if X.sanity(ds, pool):
            continue
        s_fi = pool.F[:, 0]
        s_lr = (((pool.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"]) + lrp["intercept"])
        s_cnn = X.cnn_scores(frozen, accs, table)
        s_pf = lr_score(sc, lr, pool.F[:, cols])
        gate = pool.F[:, 1]
        tabs = {}
        for name, cfgs_b in frozen["lab"].items():
            if not name.startswith("D") or "%g" % C.MAIN_BUDGET not in cfgs_b:
                continue
            s = s_lr if name.startswith("D5") else (s_cnn if name.startswith("D6") else s_fi)
            if s is None:
                continue
            cue = S.apply(pool, s, gate, cfgs_b["%g" % C.MAIN_BUDGET])
            tab, r = per_subject(pool, cue)
            tabs[name] = tab
            ci, fi = boot(tab, rng)
            ci_rows.append({"set": "external", "dataset": ds, "detector": name, "coverage_%": r["coverage_%"],
                            "cov_lo": ci[0], "cov_hi": ci[1], "fa_per_h": r["fa_per_h"], "fa_lo": fi[0], "fa_hi": fi[1],
                            "false_on_%": r["false_on_%"]})
        cue = S.apply(pool, s_pf, None, pf_cfg)
        tab, r = per_subject(pool, cue)
        tabs["D5b_LR_posture_free"] = tab
        ci, fi = boot(tab, rng)
        ci_rows.append({"set": "external (post hoc)", "dataset": ds, "detector": "D5b_LR_posture_free",
                        "coverage_%": r["coverage_%"], "cov_lo": ci[0], "cov_hi": ci[1], "fa_per_h": r["fa_per_h"],
                        "fa_lo": fi[0], "fa_hi": fi[1], "false_on_%": r["false_on_%"]})
        for a, b in [("D3_FI_gate_smooth_hyst", "D0_FI"), ("D3_FI_gate_smooth_hyst", "D5_LR_hyst"),
                     ("D3_FI_gate_smooth_hyst", "D6_CNN_hyst"), ("D3_FI_gate_smooth_hyst", "D5b_LR_posture_free")]:
            if a in tabs and b in tabs:
                lo, hi = boot_diff(tabs[a], tabs[b], rng)
                diff_rows.append({"set": "external", "dataset": ds, "comparison": "%s - %s" % (a, b), "lo": lo, "hi": hi})
        for k, s in [("FI", s_fi), ("LR", s_lr), ("LR_posture_free", s_pf), ("CNN", s_cnn)]:
            if s is None:
                continue
            auc_rows.append({"set": "external", "dataset": ds, "score": k, "auc": roc_auc_score(pool.fog, s)})
            r = matched(pool, s)
            if r:
                match_rows.append({"set": "external", "dataset": ds, "score": k, **r})
        print("external", ds, "done", flush=True)

    pd.set_option("display.width", 220)
    for name, rows in [("bootstrap_ci", ci_rows), ("bootstrap_differences", diff_rows),
                       ("matched_false_cue_5pct", match_rows), ("window_auc", auc_rows)]:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, name + ".csv"), index=False)
        print("\n" + name)
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
