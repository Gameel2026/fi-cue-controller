"""Post hoc checks requested in the final review (no frozen setting is changed).

B1) Cue precision against chance. For each test set and frozen detector, cue precision
    (share of cue starts whose run reaches an episode or its lead-in/tail) is compared with
    its circular-shift chance level (same 1,000 shifts and seed as the coverage chance).
B2) Frozen CNN (D6) on the held-out contest test recordings (lab and home), with the frozen
    D6 thresholds, coverage/timely chance, bootstrap CI and cue precision.

Outputs: results_fi/posthoc/precision_chance.csv and cnn_contest_test.csv
"""
import os
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K
import fi_select as S
from s10_posthoc import per_subject, boot
from s19_precision_gate_ap import precision

SETS = [("prepared_fi_test", "kaggle_test_tdcsfog", "lab"),
        ("prepared_fi_test", "kaggle_test_defog", "home"),
        ("prepared_fi_external", "fogstar", "lab"),
        ("prepared_fi_external", "multimodal", "lab"),
        ("prepared_fi_external", "stanford", "lab")]
V1 = ["D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest", "D5_LR_hyst", "D6_CNN_hyst"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def build(folder, g):
    Fs, fog, sc, rec, subj, accs, table = [], [], [], [], [], [], []
    for i, r in enumerate(g.itertuples()):
        z = np.load(os.path.join(folder, r.rec + ".npz"))
        a = z["acc"].astype(np.float64)
        F, idx = K.hop_features(a)
        Fs.append(F)
        fog.append(z["fog"][idx])
        sc.append(z["score"][idx].astype(bool))
        rec.append(np.full(len(idx), r.rec, dtype=object))
        subj.append(np.full(len(idx), str(r.subject), dtype=object))
        accs.append(np.concatenate([np.repeat(a[:1], C.WIN_LONG - 1, axis=0), a], axis=0).astype(np.float32))
        table.append(np.stack([np.full(len(idx), i), idx], axis=1))
    n = sum(len(f) for f in Fs)
    pool = K.Pool(np.concatenate(Fs), np.concatenate(fog), np.concatenate(sc), np.full(n, -1, np.int16),
                  np.full(n, -1, np.int16), np.concatenate(rec).astype(str), np.concatenate(subj).astype(str))
    return pool, accs, np.concatenate(table)


def cnn_scores(v1, ctx, accs, table):
    import torch
    import s4_cnn as N
    info = v1[ctx]["cnn_weights"]
    path = os.path.join(C.OUT, "frozen", info["file"])
    if sha256(path) != info["sha256"]:
        raise RuntimeError("CNN weights do not match the frozen hash")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = N.Net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    return N.predict(net, accs, table, dev)


def precision_chance(pool, cue, n_shifts, seed):
    """circular shift within each recording, same scheme and seed as fi_core.surrogate"""
    rng = np.random.RandomState(seed)
    starts = np.where(pool.rec_first)[0]
    ends = np.concatenate([starts[1:], [len(cue)]])
    out = []
    for _ in range(n_shifts):
        sh = np.empty_like(cue)
        for a, b in zip(starts, ends):
            L = b - a
            sh[a:b] = np.roll(cue[a:b], rng.randint(1, L) if L > 1 else 0)
        out.append(precision(pool, sh)[0])
    return np.array(out)


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
    prec_rows, cnn_rows = [], []
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
        cues = {}
        for name in V1:
            s = s_lr if name.startswith("D5") else (p_cnn if name.startswith("D6") else pool.F[:, 0])
            cues[name] = (pool, S.apply(pool, s, pool.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET]))
        if use_v2:
            X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
            p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
            cues["V2_GBT_hyst"] = (p2, S.apply(p2, v2_score(model, X), None, v2[ctx]["%g" % C.MAIN_BUDGET]))
        for name, (pl, cue) in cues.items():
            pr, sph = precision(pl, cue)
            ch = precision_chance(pl, cue, C.N_SHIFTS, C.SEED)
            prec_rows.append(dict(dataset=ds, detector=name, cue_precision=pr, precision_chance=np.nanmean(ch),
                                  p_precision=(1 + (ch >= pr).sum()) / (C.N_SHIFTS + 1.0), cue_starts_per_h=sph))
        if folder == "prepared_fi_test":
            cue = cues["D6_CNN_hyst"][1]
            tab, r = per_subject(pool, cue)
            ci, fa = boot(tab, rng)
            cs, ts = K.surrogate(pool, cue, C.N_SHIFTS, C.SEED)
            r.update(dataset=ds, detector="D6_CNN_hyst", subjects=len(np.unique(pool.subject)),
                     cov_lo=ci[0], cov_hi=ci[1], fa_lo=fa[0], fa_hi=fa[1],
                     coverage_chance=cs.mean(), p_coverage=(1 + (cs >= r["coverage_%"]).sum()) / (C.N_SHIFTS + 1.0),
                     timely_chance=ts.mean(), p_timely=(1 + (ts >= r["timely_%"]).sum()) / (C.N_SHIFTS + 1.0))
            cnn_rows.append(r)
        print(ds, "done", flush=True)
    a = pd.DataFrame(prec_rows)
    b = pd.DataFrame(cnn_rows)
    a.to_csv(os.path.join(out, "precision_chance.csv"), index=False)
    b.to_csv(os.path.join(out, "cnn_contest_test.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nB1 cue precision vs chance")
    print(a.round(3).to_string(index=False))
    print("\nB2 frozen CNN on the contest test recordings")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
