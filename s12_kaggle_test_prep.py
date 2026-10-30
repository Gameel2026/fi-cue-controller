import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import fi_config as C

TEST = "prepared_fi_test"
LAB = ["StartHesitation", "Turn", "Walking"]


def read_list(name):
    p = os.path.join(C.KAGGLE_ROOT, name)
    if not os.path.exists(p):
        return set()
    with open(p) as fh:
        return {ln.strip().replace(".csv", "") for ln in fh if ln.strip()}


def load_solution():
    path = os.path.join(C.KAGGLE_ROOT, "official solution.csv")
    head = pd.read_csv(path, nrows=5)
    print("solution columns:", list(head.columns))
    labs = [c for c in LAB if c in head.columns]
    usage = "Usage" if "Usage" in head.columns else None
    idcol = "Id" if "Id" in head.columns else head.columns[0]
    sol = {}
    for k, ch in enumerate(pd.read_csv(path, chunksize=2000000)):
        parts = ch[idcol].astype(str).str.rsplit("_", n=1)
        ser = parts.str[0].to_numpy()
        t = parts.str[1].astype(int).to_numpy()
        fog = (ch[labs].to_numpy().max(axis=1) > 0)
        ign = ch[usage].astype(str).str.lower().eq("ignored").to_numpy() if usage else np.zeros(len(ch), bool)
        df = pd.DataFrame({"s": ser, "t": t, "fog": fog, "ign": ign})
        for s, g in df.groupby("s"):
            sol.setdefault(s, []).append(g[["t", "fog", "ign"]].to_numpy())
        print("solution chunk", k + 1, flush=True)
    return {s: np.concatenate(v) for s, v in sol.items()}


def main():
    os.makedirs(TEST, exist_ok=True)
    overlap = read_list("overlapping public test files.txt") | read_list("overlapping private test files.txt")
    meta = pd.concat([pd.read_csv(os.path.join(C.KAGGLE_ROOT, "tdcsfog_metadata.csv")),
                      pd.read_csv(os.path.join(C.KAGGLE_ROOT, "defog_metadata.csv"))])
    subj = dict(zip(meta["Id"], meta["Subject"]))
    sol = load_solution()
    rows, skipped = [], {"overlap": 0, "no_labels": 0}
    for split in ["public", "private"]:
        for ds, fs, up, down in [("tdcsfog", 128, 1, 2), ("defog", 100, 16, 25)]:
            for f in sorted(glob.glob(os.path.join(C.KAGGLE_ROOT, split, ds, "*.csv"))):
                rid = os.path.splitext(os.path.basename(f))[0]
                if rid in overlap:
                    skipped["overlap"] += 1
                    continue
                if rid not in sol:
                    skipped["no_labels"] += 1
                    continue
                d = pd.read_csv(f)
                acc = d[["AccV", "AccML", "AccAP"]].to_numpy(float)
                if ds == "tdcsfog":
                    acc = acc / C.G
                lab = np.zeros(len(d), bool)
                ign = np.ones(len(d), bool)
                s = sol[rid]
                ok = (s[:, 0] >= 0) & (s[:, 0] < len(d))
                lab[s[ok, 0].astype(int)] = s[ok, 1].astype(bool)
                ign[s[ok, 0].astype(int)] = s[ok, 2].astype(bool)
                x = resample_poly(acc, up, down, axis=0).astype(np.float32)
                idx = np.minimum(np.round(np.arange(len(x)) * fs / C.FS).astype(int), len(d) - 1)
                n = len(x)
                np.savez_compressed(os.path.join(TEST, rid + ".npz"), acc=x, fog=lab[idx].astype(np.uint8),
                                    score=~ign[idx], ev=np.full(n, -1, np.int16), task=np.full(n, -1, np.int16))
                rows.append({"rec": rid, "dataset": "kaggle_test_" + ds, "split": split,
                             "subject": subj.get(rid, rid), "n": n, "minutes": n / C.FS / 60.0,
                             "scored_min": (~ign[idx]).sum() / C.FS / 60.0, "fog_min": lab[idx][~ign[idx]].sum() / C.FS / 60.0})
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TEST, "recordings.csv"), index=False)
    print("skipped:", skipped)
    print(r.groupby("dataset").agg(files=("rec", "count"), subjects=("subject", "nunique"),
                                   scored_h=("scored_min", lambda v: v.sum() / 60), fog_min=("fog_min", "sum")).round(2))


if __name__ == "__main__":
    main()
