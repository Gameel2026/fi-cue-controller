import os
import sys
import glob
import json
import time
import hashlib
import numpy as np
import fi_config as C
import fi_core as K
import fi_select as S
from s3_ladder import LADDER, fit_lr, lr_score

CONTEXTS = {"lab": ["tdcsfog", "daphnet"], "home": ["defog"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def load_context(names):
    parts = []
    for ds in names:
        p = os.path.join(C.PREP, "features_%s.npz" % ds)
        if os.path.exists(p):
            parts.append(np.load(p, allow_pickle=True))
    if not parts:
        return None
    cat = lambda k: np.concatenate([z[k] for z in parts])
    return K.Pool(cat("F"), cat("fog"), cat("score").astype(bool), cat("ev"), cat("task"), cat("rec"), cat("subject"))


def freeze_cnn(ctx, names, pool, frozen, out):
    import torch
    import s4_cnn as N
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs, tables = [], []
    for ds in names:
        if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
            continue
        a, t = N.load_windows(ds)
        t = t.copy()
        t[:, 0] += len(accs)
        accs += a
        tables.append(t)
    table = np.concatenate(tables)
    assert len(table) == len(pool.rec)
    idx = np.where(pool.score)[0]
    net = N.train(accs, table, idx, pool.fog.astype(np.int64), dev, C.SEED)
    p = N.predict(net, accs, table, dev)
    wpath = os.path.join(out, "cnn_%s.pt" % ctx)
    torch.save(net.state_dict(), wpath)
    cfgs = S.configs_for("hyst", p, None, pool.score)
    rows = S.run_grid(pool, p, None, cfgs)
    frozen[ctx]["D6_CNN_hyst"] = {"%g" % b: S.choose(rows, cfgs, b) for b in C.FA_BUDGETS}
    frozen[ctx]["cnn_weights"] = {"file": os.path.basename(wpath), "sha256": sha256(wpath)}


def main(with_cnn):
    out = C.ensure(os.path.join(C.OUT, "frozen"))
    frozen = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "contexts": CONTEXTS,
              "settings": {k: getattr(C, k) for k in ["FS", "HOP", "WIN_LONG", "WIN_SHORT", "LOC_BAND", "FRZ_BAND",
                                                       "ARREST_PAST", "PRE_S", "POST_S", "FA_BUDGETS", "MAIN_BUDGET",
                                                       "SPEC_FLOOR", "SEED"]},
              "features": K.FEATURES}
    for ctx, names in CONTEXTS.items():
        t0 = time.time()
        pool = load_context(names)
        if pool is None:
            continue
        frozen[ctx] = {"datasets": names, "subjects": int(len(np.unique(pool.subject))), "hours": float(pool.hours)}
        sc, lr = fit_lr(pool, C.SEED)
        lr_params = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
                     "intercept": float(lr.intercept_[0])}
        ppath = os.path.join(out, "lr_%s.json" % ctx)
        with open(ppath, "w") as fh:
            json.dump(lr_params, fh, indent=1)
        frozen[ctx]["lr_params"] = {"file": os.path.basename(ppath), "sha256": sha256(ppath)}
        s_lr = lr_score(sc, lr, pool.F)
        fi, gate = pool.F[:, 0], pool.F[:, 1]
        base = {}
        for name, level, uses_fi in LADDER:
            s = fi if uses_fi else s_lr
            g = gate if (uses_fi and level != "plain") else None
            res = {}
            if level != "arrest":
                cfgs = S.configs_for(level, s, g, pool.score)
                rows = S.run_grid(pool, s, g, cfgs)
            for b in C.FA_BUDGETS:
                if level == "arrest":
                    cfgs = S.configs_for("arrest", s, gate, pool.score, base=base[b])
                    rows = S.run_grid(pool, s, gate, cfgs)
                cfg = S.choose(rows, cfgs, b)
                res["%g" % b] = cfg
                if level == "hyst" and uses_fi:
                    base[b] = cfg
            frozen[ctx][name] = res
        if with_cnn:
            freeze_cnn(ctx, names, pool, frozen, out)
        print(ctx, "frozen in %.1f min" % ((time.time() - t0) / 60), flush=True)
    frozen["code_sha256"] = {os.path.basename(f): sha256(f) for f in sorted(glob.glob("*.py"))}
    path = os.path.join(out, "frozen_fi.json")
    with open(path, "w") as fh:
        json.dump(frozen, fh, indent=1, default=float)
    print("written", path)
    print("sha256 of frozen_fi.json:", sha256(path))


if __name__ == "__main__":
    main("--cnn" in sys.argv)
