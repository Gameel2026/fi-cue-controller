"""Post hoc description of the frozen test results (no setting is changed):
1) per-patient coverage against each patient's own circular-shift chance level
2) cue latency: seconds from FoG onset to the first cue-on sample for the episodes that were covered
Outputs: results_fi/posthoc/patient_level.csv, patient_summary.csv, latency_episodes.csv, latency_summary.csv
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
try:
    from scipy.stats import binomtest
    def sign_p(k, n):
        return binomtest(k, n, 0.5, alternative="greater").pvalue
except ImportError:
    from scipy.stats import binom_test
    def sign_p(k, n):
        return binom_test(k, n, 0.5, alternative="greater")
import fi_config as C
import fi_core as K
import fi_select as S

SETS = [("prepared_fi_test", "kaggle_test_tdcsfog", "lab"),
        ("prepared_fi_test", "kaggle_test_defog", "home"),
        ("prepared_fi_external", "fogstar", "lab"),
        ("prepared_fi_external", "multimodal", "lab"),
        ("prepared_fi_external", "stanford", "lab")]
if os.environ.get("S15_TEST"):
    SETS = [(C.PREP, "tdcsfog", "lab"), (C.PREP, "defog", "home")]
V1 = ["D0_FI", "D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest"]
N_LAT_SHIFTS = 200


def v1_pool(folder, g):
    Fs, fog, sc, rec, subj = [], [], [], [], []
    for r in g.itertuples():
        z = np.load(os.path.join(folder, r.rec + ".npz"))
        F, idx = K.hop_features(z["acc"].astype(np.float64))
        Fs.append(F)
        fog.append(z["fog"][idx])
        sc.append(z["score"][idx])
        rec.append(np.full(len(idx), r.rec, dtype=object))
        subj.append(np.full(len(idx), str(r.subject), dtype=object))
    n = sum(len(f) for f in Fs)
    return K.Pool(np.concatenate(Fs), np.concatenate(fog), np.concatenate(sc).astype(bool), np.full(n, -1, np.int16),
                  np.full(n, -1, np.int16), np.concatenate(rec).astype(str), np.concatenate(subj).astype(str))


def latencies(pool, cue):
    """onset-to-cue latency (s) for covered episodes; negative = cue started before onset (within the 4-s lead-in)"""
    cue = cue.astype(bool)
    hop = C.hop_s()
    out = np.full(len(pool.ep_start), np.nan)
    for i, (a, b) in enumerate(zip(pool.ep_start, pool.ep_end)):
        if not cue[a:b + 1].any():
            continue
        lo = max(a - pool.pre, pool.rec_start[a])
        on = np.where(cue[lo:b + 1])[0]
        out[i] = (lo + on[0] - a) * hop
    return out


def shift_cue(pool, cue, rng):
    starts = np.where(pool.rec_first)[0]
    ends = np.concatenate([starts[1:], [len(cue)]])
    sh = np.empty_like(cue)
    for a, b in zip(starts, ends):
        L = b - a
        sh[a:b] = np.roll(cue[a:b], rng.randint(1, L) if L > 1 else 0)
    return sh


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    v1 = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    use_v2 = os.path.exists(os.path.join(C.OUT, "v2", "frozen_v2.json"))
    if use_v2:
        import fi2_features as F2
        from s11_v2_develop import score as v2_score
        v2 = json.load(open(os.path.join(C.OUT, "v2", "frozen_v2.json")))
        model = pickle.load(open(os.path.join(C.OUT, "v2", v2["model"]["file"]), "rb"))
    rng = np.random.RandomState(C.SEED)
    prow, lrow, erow = [], [], []
    for folder, ds, ctx in SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            print("missing", rp)
            continue
        recs = pd.read_csv(rp)
        g = recs[recs["dataset"] == ds]
        if len(g) == 0:
            continue
        p1 = v1_pool(folder, g)
        cues = {name: (p1, S.apply(p1, p1.F[:, 0], p1.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET])) for name in V1}
        if use_v2:
            X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
            p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
            cues["V2_GBT_hyst"] = (p2, S.apply(p2, v2_score(model, X), None, v2[ctx]["%g" % C.MAIN_BUDGET]))
        for name, (pool, cue) in cues.items():
            # 1) per patient
            for s in np.unique(pool.subject):
                m = pool.subject == s
                sp = pool.subset(m)
                if len(sp.ep_start) == 0:
                    continue
                sc_ = cue[m]
                r = K.evaluate(sp, sc_)
                cs, _ = K.surrogate(sp, sc_, C.N_SHIFTS, C.SEED)
                prow.append(dict(dataset=ds, detector=name, subject=s, episodes=r["episodes"], coverage=r["coverage_%"],
                                 chance=cs.mean(), gain=r["coverage_%"] - cs.mean(),
                                 p=(1 + (cs >= r["coverage_%"]).sum()) / (C.N_SHIFTS + 1.0),
                                 fa_per_h=r["fa_per_h"], false_on=r["false_on_%"], hours=r["hours"]))
            # 2) latency
            lat = latencies(pool, cue)
            for i, v in enumerate(lat):
                erow.append(dict(dataset=ds, detector=name, subject=pool.subject[pool.ep_start[i]],
                                 duration_s=(pool.ep_end[i] - pool.ep_start[i] + 1) * C.hop_s(), latency_s=v))
            med_ch = []
            for _ in range(N_LAT_SHIFTS):
                l2 = latencies(pool, shift_cue(pool, cue.astype(bool), rng))
                if np.isfinite(l2).any():
                    med_ch.append(np.nanmedian(l2))
            v = lat[np.isfinite(lat)]
            dur = ((pool.ep_end - pool.ep_start + 1) * C.hop_s())[np.isfinite(lat)]
            lrow.append(dict(dataset=ds, detector=name, episodes=len(lat), covered=len(v),
                             median_latency_s=np.median(v) if len(v) else np.nan,
                             q1=np.percentile(v, 25) if len(v) else np.nan, q3=np.percentile(v, 75) if len(v) else np.nan,
                             pct_on_by_onset=100 * np.mean(v <= 0) if len(v) else np.nan,
                             pct_within_1s=100 * np.mean(v <= 1) if len(v) else np.nan,
                             pct_within_2s=100 * np.mean(v <= 2) if len(v) else np.nan,
                             median_frac_of_episode=np.median(np.clip(v, 0, None) / dur) if len(v) else np.nan,
                             chance_median_latency_s=np.mean(med_ch) if med_ch else np.nan,
                             p_faster_than_chance=(1 + sum(mc <= np.median(v) for mc in med_ch)) / (len(med_ch) + 1.0) if len(v) else np.nan))
        print(ds, "done", flush=True)
    P = pd.DataFrame(prow)
    P.to_csv(os.path.join(out, "patient_level.csv"), index=False)
    pd.DataFrame(erow).to_csv(os.path.join(out, "latency_episodes.csv"), index=False)
    srow = []
    for (ds, det), g in P.groupby(["dataset", "detector"], sort=False):
        d = g.gain.values
        nz = d[d != 0]
        srow.append(dict(dataset=ds, detector=det, patients=len(g), above_chance=int((d > 0).sum()),
                         below_chance=int((d < 0).sum()), p_below_005=int((g.p < 0.05).sum()),
                         median_gain=np.median(d),
                         wilcoxon_p=wilcoxon(nz, alternative="greater").pvalue if len(nz) >= 5 else np.nan,
                         sign_test_p=sign_p(int((nz > 0).sum()), len(nz)) if len(nz) else np.nan,
                         patients_ge5_episodes=int((g.episodes >= 5).sum()),
                         above_chance_ge5=int(((g.episodes >= 5) & (d > 0)).sum())))
    Ssum = pd.DataFrame(srow)
    Ssum.to_csv(os.path.join(out, "patient_summary.csv"), index=False)
    L = pd.DataFrame(lrow)
    L.to_csv(os.path.join(out, "latency_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nPER-PATIENT SUMMARY")
    print(Ssum.round(3).to_string(index=False))
    print("\nLATENCY SUMMARY (seconds from onset; negative = cue already on before onset)")
    print(L.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
