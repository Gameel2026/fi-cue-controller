import os
import json
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import fi_config as C
import fi_core as K
import fi_select as S
import fi2_features as F2

CONTEXTS = {"lab": ["tdcsfog", "daphnet"], "home": ["defog"]}
PRE_TRAIN_S = 2.0
MAX_TRAIN = 600000
OUT2 = os.path.join(C.OUT, "v2")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def train_target(pool):
    pre = int(round(PRE_TRAIN_S / C.hop_s()))
    d = np.zeros(len(pool.rec) + 1, dtype=np.int32)
    a = np.maximum(pool.ep_start - pre, pool.rec_start[pool.ep_start])
    np.add.at(d, a, 1)
    np.add.at(d, pool.ep_end + 1, -1)
    return (np.cumsum(d)[:-1] > 0).astype(int)


def fit(X, y, w_ds, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    if len(idx) > MAX_TRAIN:
        idx = rng.choice(idx, MAX_TRAIN, replace=False)
    pos = y[idx].mean()
    sw = np.where(y[idx] == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)) * w_ds[idx]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(X[idx], y[idx], sample_weight=sw)
    return m


def score(m, X):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def load_dev():
    recs = pd.read_csv(os.path.join(C.PREP, "recordings.csv"))
    recs = recs[recs["dataset"].isin(sum(CONTEXTS.values(), []))]
    X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(recs, C.PREP)
    ds_of = dict(zip(recs["rec"], recs["dataset"]))
    ds = np.array([ds_of[r] for r in rec])
    return X, names, K.Pool(X, fog, sc, ev, task, rec, subj), ds


def main():
    out = C.ensure(OUT2)
    t0 = time.time()
    X, names, pool, ds = load_dev()
    print("features:", X.shape, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    y = train_target(pool)
    w_ds = np.ones(len(y))
    for d in np.unique(ds):
        m = (ds == d) & pool.score
        w_ds[ds == d] = 1.0 / max(m.sum(), 1) * pool.score.sum() / len(np.unique(ds))
    subjects = np.unique(pool.subject)
    rng = np.random.RandomState(C.SEED)
    perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
    groups = np.array([perm[s] for s in pool.subject])
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, groups=groups)):
        tr = tr[pool.score[tr]]
        m = fit(X[tr], y[tr], w_ds[tr], C.SEED + f)
        oof[te] = score(m, X[te])
        fold[te] = f
        print("fold", f + 1, "/", C.N_FOLDS, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    rows, chosen = [], {}
    cues = {}
    for ctx, dsets in CONTEXTS.items():
        mctx = np.isin(ds, dsets)
        for d in dsets:
            cues[d] = {b: np.zeros(len(y), bool) for b in C.FA_BUDGETS}
        for f in range(C.N_FOLDS):
            sel = mctx & (fold != f)
            tst = mctx & (fold == f)
            psel, ptst = pool.subset(sel), pool.subset(tst)
            cfgs = S.configs_for("hyst", oof[sel], None, psel.score)
            grid = S.run_grid(psel, oof[sel], None, cfgs)
            for b in C.FA_BUDGETS:
                cfg = S.choose(grid, cfgs, b)
                cue = S.apply(ptst, oof[tst], None, cfg)
                for d in dsets:
                    tmp = np.zeros(len(y), bool)
                    tmp[tst] = cue
                    cues[d][b] |= tmp & (ds == d)
    for d in np.unique(ds):
        md = ds == d
        pd_ = pool.subset(md)
        for b in C.FA_BUDGETS:
            r = K.evaluate(pd_, cues[d][b][md])
            r.update(dataset=d, detector="V2_GBT_hyst", fa_budget=b)
            if b == C.MAIN_BUDGET:
                cs, ts = K.surrogate(pd_, cues[d][b][md], 300, C.SEED)
                r.update(coverage_chance=cs.mean(), p_coverage=(1 + (cs >= r["coverage_%"]).sum()) / 301.0,
                         timely_chance=ts.mean(), p_timely=(1 + (ts >= r["timely_%"]).sum()) / 301.0)
            rows.append(r)
        ok = pd_.score
        rows.append({"dataset": d, "detector": "AUC_window_V2", "coverage_%": roc_auc_score(pd_.fog[ok], oof[md][ok])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "v2_cv_summary.csv"), index=False)
    np.savez_compressed(os.path.join(out, "v2_oof.npz"), oof=oof, fold=fold, ds=ds,
                        **{"%s__%g" % (d, b): cues[d][b] for d in cues for b in C.FA_BUDGETS})
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False), flush=True)

    print("\nfreezing v2 ...", flush=True)
    final = fit(X[pool.score], y[pool.score], w_ds[pool.score], C.SEED)
    mpath = os.path.join(out, "v2_model.pkl")
    with open(mpath, "wb") as fh:
        pickle.dump(final, fh)
    frozen = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": names, "contexts": CONTEXTS,
              "pre_train_s": PRE_TRAIN_S, "sklearn": sklearn.__version__,
              "model": {"file": "v2_model.pkl", "sha256": sha256(mpath)}}
    for ctx, dsets in CONTEXTS.items():
        mctx = np.isin(ds, dsets)
        pc = pool.subset(mctx)
        cfgs = S.configs_for("hyst", oof[mctx], None, pc.score)
        grid = S.run_grid(pc, oof[mctx], None, cfgs)
        frozen[ctx] = {"%g" % b: S.choose(grid, cfgs, b) for b in C.FA_BUDGETS}
    frozen["code_sha256"] = {f: sha256(f) for f in ["fi2_features.py", "s11_v2_develop.py", "fi_core.py", "fi_select.py"]}
    fpath = os.path.join(out, "frozen_v2.json")
    with open(fpath, "w") as fh:
        json.dump(frozen, fh, indent=1, default=float)
    print("written", fpath)
    print("sha256 of frozen_v2.json:", sha256(fpath))
    print("total %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
