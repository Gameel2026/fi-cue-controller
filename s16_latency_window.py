"""Post hoc, development data only: cue latency of D3 with the planned 4-s window vs a 2-s window.
Same patient-grouped 5-fold CV and selection rule as the main analysis. No frozen setting is changed.
Output: results_fi/posthoc/latency_window_dev.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import fi_config as C
import fi_core as K
import fi_select as S
from s14_sensitivity import pool_for
from s15_patient_latency import latencies

VARIANTS = [("4 s window (planned)", 256, (3.0, 8.0)), ("2 s window", 128, (3.0, 8.0))]


def cv_cue(pool):
    subjects = np.unique(pool.subject)
    rng = np.random.RandomState(C.SEED)
    perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
    groups = np.array([perm[s] for s in pool.subject])
    cue = np.zeros(len(pool.rec), bool)
    for tr, te in GroupKFold(n_splits=min(C.N_FOLDS, len(subjects))).split(pool.F, groups=groups):
        mtr = np.zeros(len(pool.rec), bool)
        mtr[tr] = True
        ptr, pte = pool.subset(mtr), pool.subset(~mtr)
        cfgs = S.configs_for("hyst", ptr.F[:, 0], ptr.F[:, 1], ptr.score)
        cfg = S.choose(S.run_grid(ptr, ptr.F[:, 0], ptr.F[:, 1], cfgs), cfgs, C.MAIN_BUDGET)
        cue[~mtr] = S.apply(pte, pte.F[:, 0], pte.F[:, 1], cfg)
    return cue


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    rows = []
    for name, win, frz in VARIANTS:
        for ds in C.DEV_DATASETS:
            if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
                continue
            pool = pool_for(ds, win, frz)
            cue = cv_cue(pool)
            r = K.evaluate(pool, cue)
            cs, ts = K.surrogate(pool, cue, 200, C.SEED)
            lat = latencies(pool, cue)
            v = lat[np.isfinite(lat)]
            a = v[v > 0]
            dur = (pool.ep_end - pool.ep_start + 1) * C.hop_s()
            rows.append(dict(variant=name, dataset=ds, episodes=r["episodes"], median_episode_s=np.median(dur),
                             coverage=r["coverage_%"], chance=cs.mean(), gain=r["coverage_%"] - cs.mean(),
                             fa_per_h=r["fa_per_h"], false_on=r["false_on_%"], timely=r["timely_%"], timely_chance=ts.mean(),
                             covered=len(v), on_by_onset=int((v <= 0).sum()), after_onset=len(a),
                             median_latency_after_onset_s=np.median(a) if len(a) else np.nan,
                             q1=np.percentile(a, 25) if len(a) else np.nan, q3=np.percentile(a, 75) if len(a) else np.nan,
                             pct_all_episodes_cued_within_2s=100.0 * np.sum(v <= 2) / r["episodes"]))
            print(name, ds, "coverage %.1f latency %.2f s" % (r["coverage_%"], rows[-1]["median_latency_after_onset_s"]), flush=True)
    C.WIN_LONG, C.FRZ_BAND = 256, (3.0, 8.0)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "latency_window_dev.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
