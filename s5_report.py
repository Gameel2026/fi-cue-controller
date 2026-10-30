import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fi_config as C
import fi_core as K
from s3_ladder import load_pool

GROUPS = ["TUG", "Turning", "Hotspot", "4MW", "MB", "Rest"]


def task_group(name):
    for g in GROUPS:
        if name.startswith(g):
            return g
    return "Other"


def episode_labels(pool, events):
    ev_type = dict(zip(events["ev"], events["type"]))
    ev_kin = dict(zip(events["ev"], events["kinetic"]))
    types, kins = [], []
    for a, b in zip(pool.ep_start, pool.ep_end):
        e = pool.ev[a:b + 1]
        e = e[e >= 0]
        if len(e) == 0:
            types.append("unlabelled")
            kins.append("unlabelled")
            continue
        v = np.bincount(e).argmax()
        types.append(str(ev_type.get(v, "unlabelled")))
        k = ev_kin.get(v, np.nan)
        kins.append("kinetic" if k == 1 else ("akinetic" if k == 0 else "unlabelled"))
    return np.array(types), np.array(kins)


def load_cues(ds):
    cues = {}
    for fn in ["oof_%s.npz" % ds, "oof_cnn_%s.npz" % ds]:
        p = os.path.join(C.OUT, fn)
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                if "__" in k:
                    name, b = k.split("__")
                    if float(b) == C.MAIN_BUDGET:
                        cues[name] = z[k].astype(bool)
    return dict(sorted(cues.items()))


def main(datasets, n_shifts):
    events = pd.read_csv(os.path.join(C.PREP, "events_fi.csv"))
    with open(os.path.join(C.PREP, "task_codes.json")) as fh:
        code = {v: k for k, v in json.load(fh).items()}
    by_cat, fa_task, surr = [], [], []
    for ds in datasets:
        if not os.path.exists(os.path.join(C.OUT, "oof_%s.npz" % ds)):
            continue
        pool = load_pool(ds)
        cues = load_cues(ds)
        types, kins = episode_labels(pool, events)
        tg = np.array([task_group(code[t]) if t >= 0 else "none" for t in pool.task])
        for name, cue in cues.items():
            r, covered, timely, fa_starts = K.evaluate(pool, cue, detail=True)
            for col, lab in [("type", types), ("kinetic", kins)]:
                for v in np.unique(lab):
                    m = lab == v
                    by_cat.append({"dataset": ds, "detector": name, "split": col, "category": v, "episodes": int(m.sum()),
                                   "coverage_%": 100 * covered[m].mean(), "timely_%": 100 * timely[m].mean()})
            if ds == "defog":
                fa_tg = tg[fa_starts]
                non_target = pool.score & ~pool.target
                for g in GROUPS + ["Other"]:
                    hrs = (pool.score & (tg == g)).sum() * C.hop_s() / 3600.0
                    nt = non_target & (tg == g)
                    if hrs > 0:
                        fa_task.append({"detector": name, "task_group": g, "hours": hrs,
                                        "fa_per_h": (fa_tg == g).sum() / hrs,
                                        "false_cue_time_%": 100 * cue[nt].mean() if nt.any() else np.nan})
            cov_s, tim_s = K.surrogate(pool, cue, n_shifts, C.SEED)
            surr.append({"dataset": ds, "detector": name, "coverage_%": r["coverage_%"], "coverage_chance_%": cov_s.mean(),
                         "p_coverage": (1 + (cov_s >= r["coverage_%"]).sum()) / (n_shifts + 1.0),
                         "timely_%": r["timely_%"], "timely_chance_%": tim_s.mean(),
                         "p_timely": (1 + (tim_s >= r["timely_%"]).sum()) / (n_shifts + 1.0),
                         "fa_per_h": r["fa_per_h"]})
            print(ds, name, "done", flush=True)
    a = pd.DataFrame(by_cat)
    b = pd.DataFrame(fa_task)
    c = pd.DataFrame(surr)
    a.to_csv(os.path.join(C.OUT, "failure_by_fog_type.csv"), index=False)
    b.to_csv(os.path.join(C.OUT, "failure_fa_by_task.csv"), index=False)
    c.to_csv(os.path.join(C.OUT, "chance_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print(c.round(3).to_string(index=False))
    if len(a):
        piv = a[a["split"] == "kinetic"].pivot_table(index=["dataset", "detector"], columns="category", values="coverage_%")
        print("\nCoverage by kinetic/akinetic:\n", piv.round(1))
        piv2 = a[a["split"] == "type"].pivot_table(index=["dataset", "detector"], columns="category", values="coverage_%")
        print("\nCoverage by FoG type:\n", piv2.round(1))
        for ds in a["dataset"].unique():
            p = a[(a["dataset"] == ds) & (a["split"] == "kinetic")].pivot(index="detector", columns="category", values="coverage_%")
            p.plot.bar(figsize=(8, 4))
            plt.ylabel("Episode coverage (%)")
            plt.title("%s: coverage by kinetic / akinetic FoG (FA budget %g/h)" % (ds, C.MAIN_BUDGET))
            plt.tight_layout()
            plt.savefig(os.path.join(C.OUT, "fig_kinetic_%s.png" % ds), dpi=150)
            plt.close()
    if len(b):
        print("\nFalse alarms per hour by task (defog):\n", b.pivot(index="detector", columns="task_group", values="fa_per_h").round(1))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--shifts=")]
    sh = [int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("--shifts=")]
    main(args or C.DEV_DATASETS, sh[0] if sh else C.N_SHIFTS)
