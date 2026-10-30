import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import fi_config as C

FOGSTAR_CSV = r"D:\New research fog work\FoG STAR\sensor_data.csv"
FOGSTAR_AXES = (("back_acc_x", 1.0), ("back_acc_y", 1.0), ("back_acc_z", 1.0))
MULTIMODAL_ROOT = r"D:\New research fog work\Multimodal dataset 2021\r8gmbtv7w2-3\Filtered Data"
MULTIMODAL_AXES = ((46, -1.0), (47, 1.0), (48, 1.0))
MULTIMODAL_LSB_PER_G = 8192.0
STANFORD_ROOT = r"D:\New research fog work\Stanford dataset\imu-fog-detection-main\data\raw\imus6_subjects7"
EXT = "prepared_fi_external"
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def finite_parts(acc, lab, extra, min_len):
    ok = np.isfinite(acc).all(axis=1)
    e = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= min_len:
            yield acc[a:b], lab[a:b], extra[a:b]


def to64(acc, lab, extra, fs):
    from math import gcd
    g = gcd(C.FS, int(fs))
    up, down = C.FS // g, int(fs) // g
    x = resample_poly(acc, up, down, axis=0).astype(np.float32)
    idx = np.minimum(np.round(np.arange(len(x)) * fs / C.FS).astype(int), len(lab) - 1)
    return x, lab[idx], extra[idx]


def save(rows, ds, rid, subj, acc, fog, sev):
    n = len(acc)
    name = "%s__%s" % (ds, rid)
    np.savez_compressed(os.path.join(EXT, name + ".npz"), acc=acc, fog=fog.astype(np.uint8), score=np.ones(n, bool),
                        ev=np.full(n, -1, np.int16), task=sev.astype(np.int16))
    rows.append({"rec": name, "dataset": ds, "subject": "%s_%s" % (ds, subj), "n": n, "minutes": n / C.FS / 60.0})


def fogstar(rows):
    if not os.path.exists(FOGSTAR_CSV):
        say("FoG-STAR: file not found", FOGSTAR_CSV)
        return
    d = pd.read_csv(FOGSTAR_CSV)
    say("FoG-STAR columns:", list(d.columns))
    d = d.sort_values(["subjectID", "sessionID", "timestamp"], kind="mergesort").reset_index(drop=True)
    fs = 60.0
    for (s, sess), g in d.groupby(["subjectID", "sessionID"], sort=True):
        t = g["timestamp"].to_numpy(float)
        brk = np.r_[0, np.flatnonzero((np.diff(t) > 1.5 / fs) | (np.diff(t) <= 0)
                                      | (np.diff(g["taskID"].to_numpy()) != 0)) + 1, len(g)]
        for k, (a, b) in enumerate(zip(brk[:-1], brk[1:])):
            seg = g.iloc[a:b]
            acc = np.column_stack([seg[c].to_numpy(float) * sg for c, sg in FOGSTAR_AXES])
            fog = (seg["fog"].to_numpy() == 1).astype(int)
            sev = np.nan_to_num(seg["fog_severity"].to_numpy(float), nan=0.0).astype(int) * fog
            for j, (x, l, v) in enumerate(finite_parts(acc, fog, sev, int(fs))):
                x, l, v = to64(x, l, v, fs)
                save(rows, "fogstar", "s%d_sess%d_%d_%d" % (int(s), int(sess), k, j), int(s), x, l, v)


def multimodal(rows):
    paths = sorted(glob.glob(os.path.join(MULTIMODAL_ROOT, "**", "task_*.txt"), recursive=True))
    if not paths:
        say("Multimodal: no task_*.txt files under", MULTIMODAL_ROOT)
        return
    cols = [c for c, _ in MULTIMODAL_AXES]
    skipped = 0
    for p in paths:
        rel = os.path.relpath(p, MULTIMODAL_ROOT).replace("\\", "/")
        subj = rel.split("/")[0]
        d = pd.read_csv(p, header=None, usecols=cols + [60])
        acc = np.column_stack([d[c].to_numpy(float) * sg / MULTIMODAL_LSB_PER_G for c, sg in MULTIMODAL_AXES])
        if not np.any(acc != 0):
            skipped += 1
            continue
        acc[(acc == 0).all(axis=1)] = np.nan
        fog = (d[60].to_numpy() == 1).astype(int)
        for j, (x, l, v) in enumerate(finite_parts(acc, fog, np.zeros(len(fog), int), 500)):
            x, l, v = to64(x, l, v, 500)
            save(rows, "multimodal", os.path.splitext(rel)[0].replace("/", "_") + "_%d" % j, subj, x, l, v)
    say("Multimodal: %d task files without a waist sensor skipped" % skipped)


def stanford(rows):
    files = sorted(glob.glob(os.path.join(STANFORD_ROOT, "pt*.xlsx")))
    if not files:
        say("Stanford: no pt*.xlsx files in", STANFORD_ROOT)
        return
    for i, f in enumerate(files):
        d = pd.read_excel(f, engine="openpyxl")
        cols = list(d.columns)
        lum = [c for c in cols if "lumbar" in str(c).lower()]
        if i == 0:
            say("Stanford columns:", cols)
            say("Stanford lumbar columns used (first three = accelerometer):", lum[:3])
        acc = d[lum[:3]].to_numpy(float) / C.G
        vert = int(np.argmax(np.abs(np.nanmean(acc, axis=0))))
        sign = -1.0 if np.nanmean(acc[:, vert]) > 0 else 1.0
        others = [k for k in range(3) if k != vert]
        acc = np.column_stack([acc[:, vert] * sign, acc[:, others[0]], acc[:, others[1]]])
        fog = (d["freeze_label"].to_numpy() == 1).astype(int)
        base = os.path.splitext(os.path.basename(f))[0]
        for j, (x, l, v) in enumerate(finite_parts(acc, fog, np.zeros(len(fog), int), 128)):
            x, l, v = to64(x, l, v, 128)
            save(rows, "stanford", base + "_%d" % j, base.split("_")[0], x, l, v)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print("stanford", i + 1, "/", len(files), flush=True)


def review(rows):
    r = pd.DataFrame(rows)
    for ds, g in r.groupby("dataset"):
        eps, durs, means, sev = 0, [], [], {}
        for rid in g["rec"]:
            z = np.load(os.path.join(EXT, rid + ".npz"))
            fog = z["fog"].astype(int)
            e = np.flatnonzero(np.diff(np.r_[0, fog, 0]))
            eps += len(e) // 2
            durs += list((e[1::2] - e[::2]) / C.FS)
            means.append(z["acc"].mean(axis=0))
            for a, b in zip(e[::2], e[1::2]):
                v = np.bincount(z["task"][a:b].clip(0)).argmax()
                sev[int(v)] = sev.get(int(v), 0) + 1
        m = np.mean(means, axis=0)
        say("\n== %s ==" % ds)
        say("recordings:", len(g), "| subjects:", g["subject"].nunique(), "| hours: %.2f" % (g["minutes"].sum() / 60))
        say("FoG episodes:", eps, "| median duration (s): %.1f" % (np.median(durs) if durs else np.nan))
        say("mean acceleration per axis [V, ML, AP] in g (all samples):", np.round(m, 3).tolist())
        if ds == "fogstar":
            say("episodes by severity code (1 shuffling, 2 trembling, 3 akinesia, 0 none):", dict(sorted(sev.items())))


def main():
    os.makedirs(EXT, exist_ok=True)
    rows = []
    for fn in [fogstar, multimodal, stanford]:
        try:
            fn(rows)
        except Exception as e:
            say("ERROR in", fn.__name__, ":", repr(e))
    pd.DataFrame(rows).to_csv(os.path.join(EXT, "recordings.csv"), index=False)
    review(rows)
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, "external_review.txt"), "w") as fh:
        fh.write("\n".join(LINES))
    print("\nsaved", os.path.join(C.OUT, "external_review.txt"))


if __name__ == "__main__":
    main()
