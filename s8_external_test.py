import os
import json
import hashlib
import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K
import fi_select as S

EXT = "prepared_fi_external"
FROZEN = os.path.join(C.OUT, "frozen", "frozen_fi.json")
CONTEXT = "lab"
SEVERITY = {1: "shuffling", 2: "trembling", 3: "akinesia"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def build_pool(g):
    Fs, fog, sev, rec, subj, accs, table = [], [], [], [], [], [], []
    for i, r in enumerate(g.itertuples()):
        z = np.load(os.path.join(EXT, r.rec + ".npz"))
        a = z["acc"].astype(np.float64)
        F, idx = K.hop_features(a)
        Fs.append(F)
        fog.append(z["fog"][idx])
        sev.append(z["task"][idx])
        rec.append(np.full(len(idx), r.rec, dtype=object))
        subj.append(np.full(len(idx), r.subject, dtype=object))
        accs.append(np.concatenate([np.repeat(a[:1], C.WIN_LONG - 1, axis=0), a], axis=0).astype(np.float32))
        table.append(np.stack([np.full(len(idx), i), idx], axis=1))
    n = sum(len(f) for f in Fs)
    pool = K.Pool(np.concatenate(Fs), np.concatenate(fog), np.ones(n, bool), np.full(n, -1, np.int16),
                  np.concatenate(sev), np.concatenate(rec).astype(str), np.concatenate(subj).astype(str))
    return pool, accs, np.concatenate(table)


def sanity(ds, pool):
    v = pool.F[:, K.FEATURES.index("meanV")]
    problems = []
    if not -1.3 < np.median(v) < -0.7:
        problems.append("median vertical acceleration %.2f g is not close to -1 g" % np.median(v))
    if len(pool.ep_start) == 0:
        problems.append("no FoG episodes")
    if pool.hours < 0.1:
        problems.append("less than 0.1 h of data")
    return problems


def cnn_scores(frozen, accs, table):
    info = frozen[CONTEXT].get("cnn_weights")
    if info is None:
        return None
    path = os.path.join(C.OUT, "frozen", info["file"])
    try:
        import torch
        import s4_cnn as N
    except Exception:
        print("torch not available: D6 skipped")
        return None
    if sha256(path) != info["sha256"]:
        raise RuntimeError("CNN weights do not match the frozen hash")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = N.Net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    return N.predict(net, accs, table, dev)


def main():
    with open(FROZEN) as fh:
        frozen = json.load(fh)
    print("frozen configuration:", FROZEN, "sha256", sha256(FROZEN), "created", frozen["created"])
    lr_path = os.path.join(C.OUT, "frozen", frozen[CONTEXT]["lr_params"]["file"])
    if sha256(lr_path) != frozen[CONTEXT]["lr_params"]["sha256"]:
        raise RuntimeError("LR parameters do not match the frozen hash")
    with open(lr_path) as fh:
        lrp = json.load(fh)
    recs = pd.read_csv(os.path.join(EXT, "recordings.csv"))
    res, by_sev, chance = [], [], []
    for ds, g in recs.groupby("dataset"):
        pool, accs, table = build_pool(g)
        problems = sanity(ds, pool)
        if problems:
            print("\n%s NOT TESTED:" % ds, "; ".join(problems))
            res.append({"dataset": ds, "detector": "NOT TESTED: " + "; ".join(problems)})
            continue
        scores = {"FI": pool.F[:, 0],
                  "LR": (((pool.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"])
                         + lrp["intercept"]).astype(np.float32)}
        p_cnn = cnn_scores(frozen, accs, table)
        gate = pool.F[:, 1]
        sev_ep = np.array([np.bincount(pool.task[a:b + 1].clip(0)).argmax() for a, b in zip(pool.ep_start, pool.ep_end)])
        for name in [k for k in frozen[CONTEXT] if k.startswith("D")]:
            if name.startswith("D5"):
                s = scores["LR"]
            elif name.startswith("D6"):
                if p_cnn is None:
                    continue
                s = p_cnn
            else:
                s = scores["FI"]
            for b, cfg in frozen[CONTEXT][name].items():
                cue = S.apply(pool, s, gate, cfg)
                r, covered, timely, _ = K.evaluate(pool, cue, detail=True)
                r.update(dataset=ds, detector=name, fa_budget=float(b))
                res.append(r)
                if float(b) == C.MAIN_BUDGET:
                    cs, ts = K.surrogate(pool, cue, C.N_SHIFTS, C.SEED)
                    chance.append({"dataset": ds, "detector": name, "coverage_%": r["coverage_%"],
                                   "coverage_chance_%": cs.mean(),
                                   "p_coverage": (1 + (cs >= r["coverage_%"]).sum()) / (C.N_SHIFTS + 1.0),
                                   "timely_%": r["timely_%"], "timely_chance_%": ts.mean(),
                                   "p_timely": (1 + (ts >= r["timely_%"]).sum()) / (C.N_SHIFTS + 1.0),
                                   "fa_per_h": r["fa_per_h"]})
                    if ds == "fogstar":
                        for code, lab in SEVERITY.items():
                            m = sev_ep == code
                            if m.any():
                                by_sev.append({"detector": name, "severity": lab, "episodes": int(m.sum()),
                                               "coverage_%": 100 * covered[m].mean(), "timely_%": 100 * timely[m].mean()})
            print(ds, name, "done", flush=True)
    pd.DataFrame(res).to_csv(os.path.join(C.OUT, "external_results.csv"), index=False)
    pd.DataFrame(chance).to_csv(os.path.join(C.OUT, "external_chance.csv"), index=False)
    pd.DataFrame(by_sev).to_csv(os.path.join(C.OUT, "external_fogstar_by_severity.csv"), index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(chance).round(3).to_string(index=False))
    if by_sev:
        print(pd.DataFrame(by_sev).pivot(index="detector", columns="severity", values="coverage_%").round(1))


if __name__ == "__main__":
    main()
