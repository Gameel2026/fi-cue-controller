"""Post hoc, development data only: fair comparison with tuned learned models.
Hyperparameters of the logistic regression (D5) and of the gradient-boosted trees (GBT) are tuned by
patient-grouped inner cross-validation inside each outer training fold (criterion: window AUC on the
inner validation patients). Thresholds then follow the same selection rule as all other controllers.
The default (planned) hyperparameters are rerun through the identical pipeline for comparison.
Output: results_fi/posthoc/nested_tuning_dev.csv and nested_tuning_choices.csv
"""
import os
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import fi_config as C
import fi_core as K
import fi_select as S
from s3_ladder import load_pool
from s11_v2_develop import load_dev, train_target, score as gbt_score, CONTEXTS, MAX_TRAIN

LR_GRID = [0.01, 0.1, 1.0, 10.0]
LR_DEFAULT = 1.0
GBT_GRID = [dict(learning_rate=lr, max_leaf_nodes=nl, min_samples_leaf=200) for lr, nl in itertools.product([0.05, 0.1], [15, 31, 63])]
GBT_DEFAULT = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200)
INNER_FOLDS = 3
INNER_MAX = 150000
N_SH = 200


def groups_of(subject):
    subjects = np.unique(subject)
    rng = np.random.RandomState(C.SEED)
    perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
    return np.array([perm[s] for s in subject])


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


# ---------------- logistic regression ----------------
def fit_lr(X, y, Cval, seed, cap_pos=150000, cap_neg=300000):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.concatenate([rng.choice(pos, min(len(pos), cap_pos), replace=False),
                          rng.choice(neg, min(len(neg), cap_neg), replace=False)])
    sc = StandardScaler().fit(X[idx])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=Cval).fit(sc.transform(X[idx]), y[idx])
    return lambda Z: lr.decision_function(sc.transform(Z)).astype(np.float32)


def tune_lr(X, y, g, seed):
    res = {}
    for Cval in LR_GRID:
        aucs = []
        for f, (a, b) in enumerate(GroupKFold(INNER_FOLDS).split(X, groups=g)):
            fn = fit_lr(X[a], y[a], Cval, seed + f, 50000, 100000)
            aucs.append(safe_auc(y[b], fn(X[b])))
        res[Cval] = np.nanmean(aucs)
    return max(res, key=res.get), res


def run_lr(rows, choices):
    for ds in C.DEV_DATASETS:
        if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
            continue
        pool = load_pool(ds)
        groups = groups_of(pool.subject)
        cues = {"default": np.zeros(len(pool.rec), bool), "tuned": np.zeros(len(pool.rec), bool)}
        oof = {"default": np.zeros(len(pool.rec), np.float32), "tuned": np.zeros(len(pool.rec), np.float32)}
        for f, (tr, te) in enumerate(GroupKFold(min(C.N_FOLDS, len(np.unique(groups)))).split(pool.F, groups=groups)):
            mtr = np.zeros(len(pool.rec), bool)
            mtr[tr] = True
            ptr, pte = pool.subset(mtr), pool.subset(~mtr)
            m = ptr.score
            Xtr, ytr, gtr = ptr.F[m], ptr.fog[m].astype(int), groups[mtr][m]
            best, res = tune_lr(Xtr, ytr, gtr, C.SEED + 10 * f)
            choices.append(dict(model="D5 LR", dataset=ds, fold=f, chosen="C=%g" % best,
                                **{"inner_auc_C=%g" % k: v for k, v in res.items()}))
            for name, Cval in (("default", LR_DEFAULT), ("tuned", best)):
                fn = fit_lr(Xtr, ytr, Cval, C.SEED + f)
                s_tr, s_te = fn(ptr.F), fn(pte.F)
                cfgs = S.configs_for("hyst", s_tr, None, ptr.score)
                cfg = S.choose(S.run_grid(ptr, s_tr, None, cfgs), cfgs, C.MAIN_BUDGET)
                cues[name][~mtr] = S.apply(pte, s_te, None, cfg)
                oof[name][~mtr] = s_te
            print("LR", ds, "fold", f, "chosen C", best, flush=True)
        for name in cues:
            add_row(rows, "D5 LR", name, ds, pool, cues[name], oof[name], pool.fog)


# ---------------- gradient-boosted trees ----------------
def fit_gbt(X, y, w, params, seed, cap):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    pos = y[idx].mean()
    sw = np.where(y[idx] == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)) * w[idx]
    m = HistGradientBoostingClassifier(max_iter=400, l2_regularization=1.0, random_state=seed, **params)
    return m.fit(X[idx], y[idx], sample_weight=sw)


def run_gbt(rows, choices):
    t0 = time.time()
    X, names, pool, ds = load_dev()
    y = train_target(pool)
    w_ds = np.ones(len(y))
    for d in np.unique(ds):
        m = (ds == d) & pool.score
        w_ds[ds == d] = 1.0 / max(m.sum(), 1) * pool.score.sum() / len(np.unique(ds))
    groups = groups_of(pool.subject)
    oof = {"default": np.full(len(y), np.nan, np.float32), "tuned": np.full(len(y), np.nan, np.float32)}
    fold = np.full(len(y), -1, np.int8)
    for f, (tr, te) in enumerate(GroupKFold(C.N_FOLDS).split(X, groups=groups)):
        tr = tr[pool.score[tr]]
        res = {}
        for i, p in enumerate(GBT_GRID):
            aucs = []
            for j, (a, b) in enumerate(GroupKFold(INNER_FOLDS).split(X[tr], groups=groups[tr])):
                m = fit_gbt(X[tr][a], y[tr][a], w_ds[tr][a], p, C.SEED + 100 * f + j, INNER_MAX)
                bb = b if len(b) <= 300000 else np.random.RandomState(j).choice(b, 300000, replace=False)
                aucs.append(safe_auc(y[tr][bb], gbt_score(m, X[tr][bb])))
            res[i] = np.nanmean(aucs)
            print("GBT fold", f, "config", i, "inner AUC %.4f" % res[i], "%.1f min" % ((time.time() - t0) / 60), flush=True)
        best = max(res, key=res.get)
        choices.append(dict(model="GBT", dataset="all", fold=f, chosen=str(GBT_GRID[best]),
                            **{"inner_auc_%s" % str(GBT_GRID[i]): v for i, v in res.items()}))
        for name, p in (("default", GBT_DEFAULT), ("tuned", GBT_GRID[best])):
            m = fit_gbt(X[tr], y[tr], w_ds[tr], p, C.SEED + f, MAX_TRAIN)
            oof[name][te] = gbt_score(m, X[te])
        fold[te] = f
    for name in oof:
        cues = np.zeros(len(y), bool)
        for ctx, dsets in CONTEXTS.items():
            mctx = np.isin(ds, dsets)
            for f in range(C.N_FOLDS):
                sel, tst = mctx & (fold != f), mctx & (fold == f)
                psel, ptst = pool.subset(sel), pool.subset(tst)
                cfgs = S.configs_for("hyst", oof[name][sel], None, psel.score)
                cfg = S.choose(S.run_grid(psel, oof[name][sel], None, cfgs), cfgs, C.MAIN_BUDGET)
                cues[tst] = S.apply(ptst, oof[name][tst], None, cfg)
        for d in np.unique(ds):
            md = ds == d
            add_row(rows, "GBT", name, d, pool.subset(md), cues[md], oof[name][md], pool.fog[md])


def add_row(rows, model, version, ds, pool, cue, s, fog):
    r = K.evaluate(pool, cue)
    cs, _ = K.surrogate(pool, cue, N_SH, C.SEED)
    ok = pool.score
    rows.append(dict(model=model, version=version, dataset=ds, coverage=r["coverage_%"], chance=cs.mean(),
                     gain=r["coverage_%"] - cs.mean(), fa_per_h=r["fa_per_h"], false_on=r["false_on_%"],
                     timely=r["timely_%"], window_auc=safe_auc(fog[ok].astype(int), s[ok])))
    print(rows[-1], flush=True)


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    rows, choices = [], []
    run_lr(rows, choices)
    pd.DataFrame(rows).to_csv(os.path.join(out, "nested_tuning_dev.csv"), index=False)
    run_gbt(rows, choices)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "nested_tuning_dev.csv"), index=False)
    pd.DataFrame(choices).to_csv(os.path.join(out, "nested_tuning_choices.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
