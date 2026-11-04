import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import fi_config as C
import fi_core as K
import fi_select as S

VARIANTS = [("main: 4 s window, 3-8 Hz", 256, (3.0, 8.0)),
            ("2 s window", 128, (3.0, 8.0)),
            ("6 s window", 384, (3.0, 8.0)),
            ("freeze band 3-10 Hz", 256, (3.0, 10.0)),
            ("freeze band 3.5-8 Hz", 256, (3.5, 8.0))]


def pool_for(ds, win, frz):
    C.WIN_LONG, C.FRZ_BAND = win, frz
    recs = pd.read_csv(os.path.join(C.PREP, "recordings.csv"))
    recs = recs[recs["dataset"] == ds]
    Fs, fog, sc, rec, subj = [], [], [], [], []
    for r in recs.itertuples():
        z = np.load(os.path.join(C.PREP, r.rec + ".npz"))
        F, idx = K.hop_features(z["acc"].astype(np.float64))
        Fs.append(F[:, :2])
        fog.append(z["fog"][idx])
        sc.append(z["score"][idx])
        rec.append(np.full(len(idx), r.rec, dtype=object))
        subj.append(np.full(len(idx), str(r.subject), dtype=object))
    n = sum(len(f) for f in Fs)
    return K.Pool(np.concatenate(Fs), np.concatenate(fog), np.concatenate(sc).astype(bool), np.full(n, -1, np.int16),
                  np.full(n, -1, np.int16), np.concatenate(rec).astype(str), np.concatenate(subj).astype(str))


def main():
    rows = []
    for name, win, frz in VARIANTS:
        for ds in C.DEV_DATASETS:
            if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
                continue
            pool = pool_for(ds, win, frz)
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
            r = K.evaluate(pool, cue)
            cs, _ = K.surrogate(pool, cue, 200, C.SEED)
            r.update(variant=name, dataset=ds, coverage_chance=cs.mean())
            rows.append(r)
            print(name, ds, "coverage %.1f (chance %.1f) FA/h %.1f" % (r["coverage_%"], cs.mean(), r["fa_per_h"]), flush=True)
    C.WIN_LONG, C.FRZ_BAND = 256, (3.0, 8.0)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT, "sensitivity_d3.csv"), index=False)
    print(df.pivot(index="variant", columns="dataset", values="coverage_%").round(1))


if __name__ == "__main__":
    main()
