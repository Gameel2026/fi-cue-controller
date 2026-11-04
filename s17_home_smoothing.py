"""Post hoc, development data only: does the smoothing length explain the cue delay at home?
D3 on DeFOG (and the lab sets for reference) with smoothing fixed at k = 1, 4 or 8 hops (0, 1, 2 s);
all other D3 settings are chosen by the usual rule inside the same patient-grouped folds.
Output: results_fi/posthoc/latency_smoothing_dev.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import fi_config as C
import fi_core as K
import fi_select as S
from s3_ladder import load_pool
from s15_patient_latency import latencies

KS = [1, 4, 8]


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    rows = []
    for ds in ["defog", "tdcsfog", "daphnet"]:
        if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
            continue
        pool = load_pool(ds)
        subjects = np.unique(pool.subject)
        rng = np.random.RandomState(C.SEED)
        perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
        groups = np.array([perm[s] for s in pool.subject])
        cues = {k: np.zeros(len(pool.rec), bool) for k in KS}
        for tr, te in GroupKFold(n_splits=min(C.N_FOLDS, len(subjects))).split(pool.F, groups=groups):
            mtr = np.zeros(len(pool.rec), bool)
            mtr[tr] = True
            ptr, pte = pool.subset(mtr), pool.subset(~mtr)
            allc = S.configs_for("hyst", ptr.F[:, 0], ptr.F[:, 1], ptr.score)
            grid = S.run_grid(ptr, ptr.F[:, 0], ptr.F[:, 1], allc)
            for k in KS:
                idx = [i for i, c in enumerate(allc) if c["k"] == k]
                cfg = S.choose([grid[i] for i in idx], [allc[i] for i in idx], C.MAIN_BUDGET)
                cues[k][~mtr] = S.apply(pte, pte.F[:, 0], pte.F[:, 1], cfg)
        dur = (pool.ep_end - pool.ep_start + 1) * C.hop_s()
        for k in KS:
            cue = cues[k]
            r = K.evaluate(pool, cue)
            cs, _ = K.surrogate(pool, cue, 200, C.SEED)
            lat = latencies(pool, cue)
            v = lat[np.isfinite(lat)]
            a = v[v > 0]
            rows.append(dict(dataset=ds, smoothing_s=(k - 1) * C.hop_s() if k > 1 else 0.0, k=k,
                             coverage=r["coverage_%"], chance=cs.mean(), gain=r["coverage_%"] - cs.mean(),
                             fa_per_h=r["fa_per_h"], false_on=r["false_on_%"], timely=r["timely_%"],
                             median_episode_s=np.median(dur), covered=len(v), after_onset=len(a),
                             median_latency_after_onset_s=np.median(a) if len(a) else np.nan,
                             q1=np.percentile(a, 25) if len(a) else np.nan, q3=np.percentile(a, 75) if len(a) else np.nan,
                             pct_all_episodes_cued_within_2s=100.0 * np.sum(v <= 2) / r["episodes"]))
            print(ds, "k", k, "coverage %.1f latency %.2f" % (r["coverage_%"], rows[-1]["median_latency_after_onset_s"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "latency_smoothing_dev.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
