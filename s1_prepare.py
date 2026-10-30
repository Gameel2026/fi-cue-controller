import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import fi_config as C

AX = ["AccV", "AccML", "AccAP"]


def nearest_index(n_new, fs_src, n_src):
    t = np.arange(n_new) / float(C.FS)
    return np.clip(np.round(t * fs_src).astype(int), 0, n_src - 1)


def resample_acc(x, up, down):
    if up == down:
        return x.astype(np.float32)
    return resample_poly(x, up, down, axis=0).astype(np.float32)


def label_intervals(n, intervals, codes, fill=-1):
    out = np.full(n, fill, dtype=np.int16)
    t = np.arange(n) / float(C.FS)
    for (a, b), c in zip(intervals, codes):
        out[(t >= a) & (t <= b)] = c
    return out


def main():
    prep = C.ensure(C.PREP)
    events = pd.read_csv(os.path.join(C.KAGGLE_ROOT, "events.csv"))
    tasks = pd.read_csv(os.path.join(C.KAGGLE_ROOT, "tasks.csv"))
    meta_t = pd.read_csv(os.path.join(C.KAGGLE_ROOT, "tdcsfog_metadata.csv"))
    meta_d = pd.read_csv(os.path.join(C.KAGGLE_ROOT, "defog_metadata.csv"))
    task_names = sorted(tasks["Task"].dropna().unique().tolist())
    task_code = {k: i for i, k in enumerate(task_names)}
    ev_rows = []
    rec_rows = []
    uid = 0

    for ds, fs_src, up, down, meta in [("tdcsfog", 128, 1, 2, meta_t), ("defog", 100, 16, 25, meta_d)]:
        files = sorted(glob.glob(os.path.join(C.KAGGLE_ROOT, "train", ds, "*.csv")))
        subj = dict(zip(meta["Id"], meta["Subject"]))
        med = dict(zip(meta["Id"], meta["Medication"]))
        for k, f in enumerate(files):
            rid = os.path.splitext(os.path.basename(f))[0]
            d = pd.read_csv(f)
            acc = d[AX].to_numpy(dtype=np.float64)
            if ds == "tdcsfog":
                acc = acc / C.G
            acc = resample_acc(acc, up, down)
            n = len(acc)
            idx = nearest_index(n, fs_src, len(d))
            fog = (d[["StartHesitation", "Turn", "Walking"]].to_numpy().max(axis=1) > 0)[idx].astype(np.uint8)
            if ds == "defog":
                valid = d["Valid"].astype(str).str.lower().eq("true").to_numpy()
                intask = d["Task"].astype(str).str.lower().eq("true").to_numpy()
                score = (valid & intask)[idx]
            else:
                score = np.ones(n, dtype=bool)
            e = events[events["Id"] == rid]
            codes = []
            for _, r in e.iterrows():
                ev_rows.append({"ev": uid, "rec": rid, "dataset": ds, "type": r["Type"] if isinstance(r["Type"], str) else "NA",
                                "kinetic": r["Kinetic"], "init_s": r["Init"], "end_s": r["Completion"]})
                codes.append(uid)
                uid += 1
            ev = label_intervals(n, list(zip(e["Init"], e["Completion"])), codes)
            tk = tasks[tasks["Id"] == rid]
            task = label_intervals(n, list(zip(tk["Begin"], tk["End"])), [task_code[t] for t in tk["Task"]])
            np.savez_compressed(os.path.join(prep, rid + ".npz"), acc=acc, fog=fog, score=score, ev=ev, task=task)
            rec_rows.append({"rec": rid, "dataset": ds, "subject": subj.get(rid, rid), "medication": med.get(rid, "NA"),
                             "n": n, "minutes": n / C.FS / 60.0, "fog_min": fog[score].sum() / C.FS / 60.0,
                             "scored_min": score.sum() / C.FS / 60.0})
            if (k + 1) % 50 == 0 or k + 1 == len(files):
                print(ds, k + 1, "/", len(files), flush=True)

    if os.path.isdir(C.DAPHNET_ROOT):
        cols = ["t", "ank_x", "ank_y", "ank_z", "th_x", "th_y", "th_z", "tr_x", "tr_y", "tr_z", "label"]
        for f in sorted(glob.glob(os.path.join(C.DAPHNET_ROOT, "S*R*.txt"))):
            rid = "daphnet_" + os.path.splitext(os.path.basename(f))[0]
            d = pd.read_csv(f, sep=r"\s+", header=None, names=cols)
            acc = np.stack([d[c].to_numpy(dtype=np.float64) * s / 1000.0 for c, s in C.DAPHNET_AXES], axis=1).astype(np.float32)
            lab = d["label"].to_numpy()
            n = len(acc)
            fog = (lab == 2).astype(np.uint8)
            score = lab != 0
            np.savez_compressed(os.path.join(prep, rid + ".npz"), acc=acc, fog=fog, score=score,
                                ev=np.full(n, -1, np.int16), task=np.full(n, -1, np.int16))
            rec_rows.append({"rec": rid, "dataset": "daphnet", "subject": rid.split("_")[1][:3], "medication": "NA",
                             "n": n, "minutes": n / C.FS / 60.0, "fog_min": fog[score].sum() / C.FS / 60.0,
                             "scored_min": score.sum() / C.FS / 60.0})
        print("daphnet done", flush=True)
    else:
        print("Daphnet folder not found, skipped:", C.DAPHNET_ROOT)

    pd.DataFrame(rec_rows).to_csv(os.path.join(prep, "recordings.csv"), index=False)
    pd.DataFrame(ev_rows).to_csv(os.path.join(prep, "events_fi.csv"), index=False)
    with open(os.path.join(prep, "task_codes.json"), "w") as fh:
        json.dump(task_code, fh, indent=1)
    r = pd.DataFrame(rec_rows)
    print(r.groupby("dataset").agg(recordings=("rec", "count"), subjects=("subject", "nunique"),
                                   scored_h=("scored_min", lambda x: x.sum() / 60), fog_min=("fog_min", "sum")).round(2))


if __name__ == "__main__":
    main()
