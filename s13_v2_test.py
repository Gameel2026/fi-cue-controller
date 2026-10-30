import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K
import fi_select as S
import fi2_features as F2
from s10_posthoc import per_subject, boot
from s11_v2_develop import score as v2_score

SETS = [("prepared_fi_test", "kaggle_test_tdcsfog", "lab", "primary"),
        ("prepared_fi_test", "kaggle_test_defog", "home", "primary"),
        ("prepared_fi_external", "fogstar", "lab", "secondary"),
        ("prepared_fi_external", "multimodal", "lab", "secondary"),
        ("prepared_fi_external", "stanford", "lab", "secondary")]
V1 = ["D0_FI", "D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest", "D5_LR_hyst"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


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


def main():
    out = C.ensure(os.path.join(C.OUT, "v2"))
    f1 = os.path.join(C.OUT, "frozen", "frozen_fi.json")
    f2 = os.path.join(C.OUT, "v2", "frozen_v2.json")
    v1 = json.load(open(f1))
    v2 = json.load(open(f2))
    print("v1 frozen sha256:", sha256(f1))
    print("v2 frozen sha256:", sha256(f2))
    mpath = os.path.join(C.OUT, "v2", v2["model"]["file"])
    if sha256(mpath) != v2["model"]["sha256"]:
        raise RuntimeError("v2 model does not match the frozen hash")
    model = pickle.load(open(mpath, "rb"))
    rng = np.random.RandomState(C.SEED)
    rows = []
    for folder, ds, ctx, role in SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            print("missing", rp)
            continue
        recs = pd.read_csv(rp)
        g = recs[recs["dataset"] == ds]
        if len(g) == 0:
            print("no recordings for", ds)
            continue
        p1 = v1_pool(folder, g)
        X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
        if names != v2["features"]:
            raise RuntimeError("feature list differs from the frozen v2 model")
        p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
        cues = {}
        lrp = json.load(open(os.path.join(C.OUT, "frozen", v1[ctx]["lr_params"]["file"])))
        s_lr = (((p1.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"]) + lrp["intercept"])
        for name in V1:
            s = s_lr if name.startswith("D5") else p1.F[:, 0]
            cues[("v1", name)] = (p1, S.apply(p1, s, p1.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET]))
        s2 = v2_score(model, X)
        cues[("v2", "V2_GBT_hyst")] = (p2, S.apply(p2, s2, None, v2[ctx]["%g" % C.MAIN_BUDGET]))
        for (ver, name), (pool, cue) in cues.items():
            tab, r = per_subject(pool, cue)
            ci, fa = boot(tab, rng)
            cs, ts = K.surrogate(pool, cue, C.N_SHIFTS, C.SEED)
            r.update(dataset=ds, role=role, version=ver, detector=name, cov_lo=ci[0], cov_hi=ci[1], fa_lo=fa[0], fa_hi=fa[1],
                     coverage_chance=cs.mean(), p_coverage=(1 + (cs >= r["coverage_%"]).sum()) / (C.N_SHIFTS + 1.0),
                     timely_chance=ts.mean(), p_timely=(1 + (ts >= r["timely_%"]).sum()) / (C.N_SHIFTS + 1.0),
                     subjects=len(np.unique(pool.subject)))
            rows.append(r)
        print(ds, "done", flush=True)
    df = pd.DataFrame(rows)
    cols = ["role", "dataset", "version", "detector", "subjects", "episodes", "coverage_%", "cov_lo", "cov_hi",
            "coverage_chance", "p_coverage", "timely_%", "timely_chance", "p_timely", "fa_per_h", "fa_lo", "fa_hi",
            "false_on_%", "hours"]
    df = df[cols]
    df.to_csv(os.path.join(out, "v2_test_results.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["hours"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
