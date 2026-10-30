import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import fi_config as C
import fi_core as K
import fi_select as S

LADDER = [("D0_FI", "plain", True), ("D1_FI_gate", "gate", True), ("D2_FI_gate_smooth", "smooth", True),
          ("D3_FI_gate_smooth_hyst", "hyst", True), ("D4_D3_plus_arrest", "arrest", True),
          ("D5_LR_hyst", "hyst", False)]


def load_pool(ds):
    z = np.load(os.path.join(C.PREP, "features_%s.npz" % ds), allow_pickle=True)
    return K.Pool(z["F"], z["fog"], z["score"].astype(bool), z["ev"], z["task"], z["rec"], z["subject"])


def fit_lr(pool, seed):
    m = pool.score
    X, y = pool.F[m], pool.fog[m].astype(int)
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pos = rng.choice(pos, min(len(pos), 150000), replace=False)
    neg = rng.choice(neg, min(len(neg), 300000), replace=False)
    idx = np.concatenate([pos, neg])
    sc = StandardScaler().fit(X[idx])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(sc.transform(X[idx]), y[idx])
    return sc, lr


def lr_score(sc, lr, F):
    return lr.decision_function(sc.transform(F)).astype(np.float32)


def main(datasets):
    out = C.ensure(C.OUT)
    summary, coefs, chosen = [], [], {}
    for ds in datasets:
        path = os.path.join(C.PREP, "features_%s.npz" % ds)
        if not os.path.exists(path):
            print("skip", ds, "(no features)")
            continue
        t0 = time.time()
        pool = load_pool(ds)
        subjects = np.unique(pool.subject)
        n_folds = min(C.N_FOLDS, len(subjects))
        rng = np.random.RandomState(C.SEED)
        perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
        groups = np.array([perm[s] for s in pool.subject])
        oof = {}
        oof_lr = np.full(len(pool.rec), np.nan, dtype=np.float32)
        fold_id = np.full(len(pool.rec), -1, dtype=np.int8)
        for f, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(pool.F, groups=groups)):
            mtr = np.zeros(len(pool.rec), bool)
            mtr[tr] = True
            ptr, pte = pool.subset(mtr), pool.subset(~mtr)
            fold_id[~mtr] = f
            sc, lr = fit_lr(ptr, C.SEED + f)
            s_lr_tr, s_lr_te = lr_score(sc, lr, ptr.F), lr_score(sc, lr, pte.F)
            oof_lr[~mtr] = s_lr_te
            coefs.append(dict(zip(K.FEATURES, lr.coef_[0] / 1.0), dataset=ds, fold=f))
            fi_tr, fi_te = ptr.F[:, 0], pte.F[:, 0]
            gate_tr, gate_te = ptr.F[:, 1], pte.F[:, 1]
            base_cfg = {}
            for name, level, uses_fi in LADDER:
                s_tr, s_te = (fi_tr, fi_te) if uses_fi else (s_lr_tr, s_lr_te)
                g_tr = gate_tr if (uses_fi and level != "plain") else None
                g_te = gate_te if (uses_fi and level != "plain") else None
                if level != "arrest":
                    cfgs = S.configs_for(level, s_tr, g_tr, ptr.score)
                    rows = S.run_grid(ptr, s_tr, g_tr, cfgs)
                for budget in C.FA_BUDGETS:
                    if level == "arrest":
                        cfgs = S.configs_for("arrest", s_tr, gate_tr, ptr.score, base=base_cfg[budget])
                        rows = S.run_grid(ptr, s_tr, gate_tr, cfgs)
                    cfg = S.choose(rows, cfgs, budget)
                    if level == "hyst" and uses_fi:
                        base_cfg[budget] = cfg
                    cue = S.apply(pte, s_te, g_te if level != "arrest" else gate_te, cfg)
                    key = "%s|%g" % (name, budget)
                    oof.setdefault(key, np.zeros(len(pool.rec), bool))[~mtr] = cue
                    chosen.setdefault(ds, {}).setdefault(key, []).append(cfg)
            print(ds, "fold", f + 1, "/", n_folds, "done in %.0f s" % (time.time() - t0), flush=True)
        for key, cue in oof.items():
            name, budget = key.split("|")
            r = K.evaluate(pool, cue)
            r.update(dataset=ds, detector=name, fa_budget=float(budget))
            summary.append(r)
        m = pool.score
        auc_fi = roc_auc_score(pool.fog[m], pool.F[m, 0])
        auc_lr = roc_auc_score(pool.fog[m], oof_lr[m])
        summary.append({"dataset": ds, "detector": "AUC_window_logFI4", "coverage_%": auc_fi})
        summary.append({"dataset": ds, "detector": "AUC_window_LR_oof", "coverage_%": auc_lr})
        np.savez_compressed(os.path.join(out, "oof_%s.npz" % ds), fold=fold_id, lr=oof_lr,
                            **{k.replace("|", "__"): v for k, v in oof.items()})
        print(ds, "finished in %.1f min" % ((time.time() - t0) / 60), flush=True)
    cols = ["dataset", "detector", "fa_budget", "episodes", "coverage_%", "timely_%", "fa_per_h", "cue_on_%", "false_on_%", "hours"]
    df = pd.DataFrame(summary)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(out, "cv_summary.csv"), index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(out, "lr_coefficients.csv"), index=False)
    with open(os.path.join(out, "chosen_configs.json"), "w") as fh:
        json.dump(chosen, fh, indent=1, default=float)
    print(df[df["fa_budget"] == C.MAIN_BUDGET].round(2).to_string(index=False))
    print(df[df["detector"].str.startswith("AUC")][["dataset", "detector", "coverage_%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or C.DEV_DATASETS)
