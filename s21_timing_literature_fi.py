"""Post hoc checks requested in the final review (no frozen setting is changed).

A) Onset-aware metrics on every test set for the frozen controllers (D0, D3, D4, D5, D6, GBT):
   - sample-level sensitivity (share of scored FoG samples with the cue on) and specificity
     (share of scored non-target samples with the cue off, i.e. 100 - false-cue time);
   - detection within L s of onset (L = 0, 1, 2 s): the cue is on at some sample between onset and
     onset + L, in a run that started no earlier than the 4-s lead-in (L = 0 equals timely activation);
   - event F1 at 2 s: harmonic mean of cue precision and detection within 2 s;
   - circular-shift chance (1,000 shifts, same seed) for detection within 2 s.
B) Untuned literature Freeze Index detectors applied without any re-tuning: FI ratio above 1.5
   (Baechlin et al. code) and above 2.3 (Moore et al. global threshold), on the 4-s FI of this study,
   with no energy gate, smoothing or hysteresis. Coverage, chance, FA/h, false-cue time and cue precision.

Outputs: results_fi/posthoc/timing_metrics.csv and literature_fi.csv
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K
import fi_select as S
from s19_precision_gate_ap import precision
from s20_precision_chance_cnn_test import SETS, build, cnn_scores

V1 = ["D0_FI", "D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest", "D5_LR_hyst", "D6_CNN_hyst"]
LAGS = [0.0, 1.0, 2.0]
LIT = {"LIT_FI_1.5_Baechlin": 1.5, "LIT_FI_2.3_Moore": 2.3}


def run_start(pool, cue):
    prev = np.concatenate([[False], cue[:-1]])
    prev[pool.rec_first] = False
    st = cue & ~prev
    rs = np.where(st, np.arange(len(cue)), 0)
    np.maximum.accumulate(rs, out=rs)
    return rs


def detect_within(pool, cue, lag_s):
    cue = cue.astype(bool)
    rs = run_start(pool, cue)
    a, b = pool.ep_start, pool.ep_end
    if len(a) == 0:
        return np.nan
    L = int(round(lag_s / C.hop_s()))
    idx = np.minimum(a[:, None] + np.arange(L + 1)[None, :], b[:, None])
    ok = cue[idx] & (rs[idx] >= (a - pool.pre)[:, None])
    return 100.0 * ok.any(axis=1).mean()


def shifted(pool, cue, rng):
    starts = np.where(pool.rec_first)[0]
    ends = np.concatenate([starts[1:], [len(cue)]])
    sh = np.empty_like(cue)
    for a, b in zip(starts, ends):
        L = b - a
        sh[a:b] = np.roll(cue[a:b], rng.randint(1, L) if L > 1 else 0)
    return sh


def timing_row(ds, name, pool, cue):
    cue = cue.astype(bool)
    fog = (pool.fog > 0) & pool.score
    sens = 100.0 * cue[fog].mean() if fog.any() else np.nan
    r = K.evaluate(pool, cue)
    spec = 100.0 - r["false_on_%"]
    prec, _ = precision(pool, cue)
    row = dict(dataset=ds, detector=name, sample_sensitivity=sens, sample_specificity=spec, cue_precision=prec)
    for L in LAGS:
        row["detect_within_%gs" % L] = detect_within(pool, cue, L)
    rec2 = row["detect_within_2s"]
    row["event_F1_2s"] = 2 * prec * rec2 / (prec + rec2) if (prec + rec2) > 0 else np.nan
    rng = np.random.RandomState(C.SEED)
    ch = np.array([detect_within(pool, shifted(pool, cue, rng), 2.0) for _ in range(C.N_SHIFTS)])
    row["detect_within_2s_chance"] = ch.mean()
    row["p_detect_within_2s"] = (1 + (ch >= rec2).sum()) / (C.N_SHIFTS + 1.0)
    return row


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    v1 = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    use_v2 = os.path.exists(os.path.join(C.OUT, "v2", "frozen_v2.json"))
    if use_v2:
        import fi2_features as F2
        from s11_v2_develop import score as v2_score
        v2 = json.load(open(os.path.join(C.OUT, "v2", "frozen_v2.json")))
        model = pickle.load(open(os.path.join(C.OUT, "v2", v2["model"]["file"]), "rb"))
    trows, lrows = [], []
    for folder, ds, ctx in SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            print("missing", rp)
            continue
        recs = pd.read_csv(rp)
        g = recs[recs["dataset"] == ds]
        if len(g) == 0:
            continue
        pool, accs, table = build(folder, g)
        lrp = json.load(open(os.path.join(C.OUT, "frozen", v1[ctx]["lr_params"]["file"])))
        s_lr = (((pool.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"]) + lrp["intercept"])
        p_cnn = cnn_scores(v1, ctx, accs, table)
        for name in V1:
            s = s_lr if name.startswith("D5") else (p_cnn if name.startswith("D6") else pool.F[:, 0])
            cue = S.apply(pool, s, pool.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET])
            trows.append(timing_row(ds, name, pool, cue))
        if use_v2:
            X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
            p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
            cue = S.apply(p2, v2_score(model, X), None, v2[ctx]["%g" % C.MAIN_BUDGET])
            trows.append(timing_row(ds, "V2_GBT_hyst", p2, cue))
        for name, thr in LIT.items():
            cue = pool.F[:, 0] > np.log(thr)
            r = K.evaluate(pool, cue)
            cs, ts = K.surrogate(pool, cue, C.N_SHIFTS, C.SEED)
            prec, _ = precision(pool, cue)
            lrows.append(dict(dataset=ds, detector=name, coverage=r["coverage_%"], coverage_chance=cs.mean(),
                              p_coverage=(1 + (cs >= r["coverage_%"]).sum()) / (C.N_SHIFTS + 1.0),
                              timely=r["timely_%"], timely_chance=ts.mean(), fa_per_h=r["fa_per_h"],
                              false_on=r["false_on_%"], cue_precision=prec))
            trows.append(timing_row(ds, name, pool, cue))
        print(ds, "done", flush=True)
    a = pd.DataFrame(trows)
    b = pd.DataFrame(lrows)
    a.to_csv(os.path.join(out, "timing_metrics.csv"), index=False)
    b.to_csv(os.path.join(out, "literature_fi.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nA) onset-aware metrics")
    print(a.round(2).to_string(index=False))
    print("\nB) untuned literature FI detectors")
    print(b.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
