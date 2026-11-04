"""Post hoc descriptions requested in review (no frozen setting is changed):
A) test sets: cue precision (share of cue starts that reach an episode or its lead-in/tail), cue starts per hour,
   and window-level average precision (AP) of each score for FoG presence, with the prevalence as the chance AP
B) development: how often the energy gate was closed during kinetic and akinetic episodes (frozen gate values)
Outputs: results_fi/posthoc/precision_ap_test.csv and gate_akinetic_dev.csv
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import fi_config as C
import fi_core as K
import fi_select as S
from s15_patient_latency import SETS, v1_pool
from s3_ladder import load_pool
from s5_report import episode_labels

DETS = ["D3_FI_gate_smooth_hyst", "D4_D3_plus_arrest", "D5_LR_hyst"]


def precision(pool, cue):
    cue = cue.astype(bool)
    cs, ce = K.run_bounds(cue, pool.rec_first)
    tcs = np.concatenate([[0], np.cumsum(pool.target)])
    hit = (tcs[ce + 1] - tcs[cs]) > 0
    keep = pool.score[cs]
    n = int(keep.sum())
    return (100.0 * hit[keep].sum() / n if n else np.nan), n / pool.hours if pool.hours else np.nan


def part_a(out):
    v1 = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    use_v2 = os.path.exists(os.path.join(C.OUT, "v2", "frozen_v2.json"))
    if use_v2:
        import fi2_features as F2
        from s11_v2_develop import score as v2_score
        v2 = json.load(open(os.path.join(C.OUT, "v2", "frozen_v2.json")))
        model = pickle.load(open(os.path.join(C.OUT, "v2", v2["model"]["file"]), "rb"))
    rows = []
    for folder, ds, ctx in SETS:
        rp = os.path.join(folder, "recordings.csv")
        if not os.path.exists(rp):
            continue
        recs = pd.read_csv(rp)
        g = recs[recs["dataset"] == ds]
        if len(g) == 0:
            continue
        p1 = v1_pool(folder, g)
        lrp = json.load(open(os.path.join(C.OUT, "frozen", v1[ctx]["lr_params"]["file"])))
        s_lr = (((p1.F - np.array(lrp["mean"])) / np.array(lrp["scale"])) @ np.array(lrp["coef"]) + lrp["intercept"])
        ok = p1.score
        y = (p1.fog[ok] > 0).astype(int)
        prev = 100.0 * y.mean()
        scores = {"FI": p1.F[ok, 0], "LR": s_lr[ok]}
        for name in DETS:
            s = s_lr if name.startswith("D5") else p1.F[:, 0]
            cue = S.apply(p1, s, p1.F[:, 1], v1[ctx][name]["%g" % C.MAIN_BUDGET])
            pr, sph = precision(p1, cue)
            rows.append(dict(dataset=ds, item=name, cue_precision=pr, cue_starts_per_h=sph))
        if use_v2:
            X, names, rec, fog, sc, ev, task, subj = F2.recordings_features(g, folder)
            p2 = K.Pool(X, fog, sc, ev, task, rec, subj)
            s2 = v2_score(model, X)
            cue = S.apply(p2, s2, None, v2[ctx]["%g" % C.MAIN_BUDGET])
            pr, sph = precision(p2, cue)
            rows.append(dict(dataset=ds, item="V2_GBT_hyst", cue_precision=pr, cue_starts_per_h=sph))
            ok2 = p2.score
            rows.append(dict(dataset=ds, item="AP_GBT", ap=100 * average_precision_score((p2.fog[ok2] > 0).astype(int), s2[ok2]),
                             prevalence=100.0 * (p2.fog[ok2] > 0).mean()))
        for k, v in scores.items():
            rows.append(dict(dataset=ds, item="AP_" + k, ap=100 * average_precision_score(y, v), prevalence=prev))
        print(ds, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "precision_ap_test.csv"), index=False)
    return df


def part_b(out):
    v1 = json.load(open(os.path.join(C.OUT, "frozen", "frozen_fi.json")))
    events = pd.read_csv(os.path.join(C.PREP, "events_fi.csv"))
    rows = []
    for ds, ctx in [("tdcsfog", "lab"), ("defog", "home")]:
        if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
            continue
        pool = load_pool(ds)
        cfg = v1[ctx]["D3_FI_gate_smooth_hyst"]["%g" % C.MAIN_BUDGET]
        closed = pool.F[:, 1] <= cfg["g"]
        _, kins = episode_labels(pool, events)
        frac = np.array([closed[a:b + 1].mean() for a, b in zip(pool.ep_start, pool.ep_end)])
        for k in ["kinetic", "akinetic"]:
            m = kins == k
            if m.sum() == 0:
                continue
            rows.append(dict(dataset=ds, motor_type=k, episodes=int(m.sum()), gate=cfg["g"],
                             pct_gate_closed_majority=100.0 * np.mean(frac[m] >= 0.5),
                             pct_gate_closed_entire=100.0 * np.mean(frac[m] >= 0.999),
                             median_closed_fraction=float(np.median(frac[m]))))
        print(ds, "gate done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "gate_akinetic_dev.csv"), index=False)
    return df


def main():
    out = C.ensure(os.path.join(C.OUT, "posthoc"))
    a = part_a(out)
    b = part_b(out)
    pd.set_option("display.width", 250)
    print("\nTEST: cue precision, cue starts per hour, window AP (prevalence = chance AP)")
    print(a.round(2).to_string(index=False))
    print("\nDEVELOPMENT: energy gate during kinetic vs akinetic episodes")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
