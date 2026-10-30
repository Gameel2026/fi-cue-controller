import os
import numpy as np
import pandas as pd
import fi_config as C
import fi_core as K


def main():
    recs = pd.read_csv(os.path.join(C.PREP, "recordings.csv"))
    for ds, g in recs.groupby("dataset"):
        Fs, fog, score, ev, task, rec, subj = [], [], [], [], [], [], []
        for k, r in enumerate(g.itertuples()):
            z = np.load(os.path.join(C.PREP, r.rec + ".npz"))
            F, idx = K.hop_features(z["acc"].astype(np.float64))
            Fs.append(F)
            fog.append(z["fog"][idx])
            score.append(z["score"][idx])
            ev.append(z["ev"][idx])
            task.append(z["task"][idx])
            rec.append(np.full(len(idx), r.rec, dtype=object))
            subj.append(np.full(len(idx), str(r.subject), dtype=object))
            if (k + 1) % 50 == 0 or k + 1 == len(g):
                print(ds, k + 1, "/", len(g), flush=True)
        out = os.path.join(C.PREP, "features_%s.npz" % ds)
        np.savez_compressed(out, F=np.concatenate(Fs), fog=np.concatenate(fog), score=np.concatenate(score),
                            ev=np.concatenate(ev), task=np.concatenate(task),
                            rec=np.concatenate(rec).astype(str), subject=np.concatenate(subj).astype(str),
                            names=np.array(K.FEATURES))
        print("saved", out, "hops:", sum(len(f) for f in Fs))


if __name__ == "__main__":
    main()
