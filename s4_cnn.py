import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import fi_config as C
import fi_core as K
import fi_select as S
from s3_ladder import load_pool

EPOCHS = 4
BATCH = 512
MAX_TRAIN = 200000


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(6, 32), nn.MaxPool1d(2), blk(32, 64), nn.MaxPool1d(2), blk(64, 64),
                               nn.AdaptiveAvgPool1d(1))
        self.h = nn.Linear(64, 1)

    def forward(self, x):
        return self.h(self.f(x).squeeze(-1)).squeeze(-1)


def load_windows(ds):
    recs = pd.read_csv(os.path.join(C.PREP, "recordings.csv"))
    recs = recs[recs["dataset"] == ds]
    accs, table = [], []
    for i, r in enumerate(recs.itertuples()):
        a = np.load(os.path.join(C.PREP, r.rec + ".npz"))["acc"].astype(np.float32)
        accs.append(np.concatenate([np.repeat(a[:1], C.WIN_LONG - 1, axis=0), a], axis=0))
        n_hops = (len(a) - 1) // C.HOP + 1
        table.append(np.stack([np.full(n_hops, i), np.arange(n_hops) * C.HOP], axis=1))
    return accs, np.concatenate(table)


def batch(accs, table, idx):
    w = np.stack([accs[r][s:s + C.WIN_LONG] for r, s in table[idx]])
    m = w.mean(axis=1, keepdims=True)
    x = np.concatenate([w - m, np.repeat(m, C.WIN_LONG, axis=1)], axis=2)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))


def predict(net, accs, table, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(table), 4096):
            out.append(net(batch(accs, table, np.arange(i, min(i + 4096, len(table)))).to(dev)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train(accs, table, idx, y, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    if len(idx) > MAX_TRAIN:
        idx = rng.choice(idx, MAX_TRAIN, replace=False)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pw = torch.tensor((y[idx] == 0).sum() / max(1, (y[idx] == 1).sum()), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(idx)
        tot = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            xb = batch(accs, table, b).to(dev)
            yb = torch.from_numpy(y[b].astype(np.float32)).to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print("   epoch", ep + 1, "loss %.4f" % (tot / len(order)), flush=True)
    return net


def main(datasets):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    rows = []
    for ds in datasets:
        if not os.path.exists(os.path.join(C.PREP, "features_%s.npz" % ds)):
            continue
        t0 = time.time()
        pool = load_pool(ds)
        accs, table = load_windows(ds)
        assert len(table) == len(pool.rec), "hop tables do not match features"
        subjects = np.unique(pool.subject)
        rng = np.random.RandomState(C.SEED)
        perm = {s: i for i, s in enumerate(rng.permutation(subjects))}
        groups = np.array([perm[s] for s in pool.subject])
        n_folds = min(C.N_FOLDS, len(subjects))
        oof_p = np.full(len(pool.rec), np.nan, dtype=np.float32)
        oof = {}
        y = pool.fog.astype(np.int64)
        for f, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(pool.F, groups=groups)):
            mtr = np.zeros(len(pool.rec), bool)
            mtr[tr] = True
            tr_idx = np.where(mtr & pool.score)[0]
            net = train(accs, table, tr_idx, y, dev, C.SEED + f)
            p = predict(net, accs, table, dev)
            oof_p[~mtr] = p[~mtr]
            ptr, pte = pool.subset(mtr), pool.subset(~mtr)
            cfgs = S.configs_for("hyst", p[mtr], None, ptr.score)
            rows = S.run_grid(ptr, p[mtr], None, cfgs)
            for budget in C.FA_BUDGETS:
                cfg = S.choose(rows, cfgs, budget)
                cue = S.apply(pte, p[~mtr], None, cfg)
                oof.setdefault("D6_CNN_hyst|%g" % budget, np.zeros(len(pool.rec), bool))[~mtr] = cue
            print(ds, "fold", f + 1, "/", n_folds, "%.0f s" % (time.time() - t0), flush=True)
        for key, cue in oof.items():
            name, budget = key.split("|")
            r = K.evaluate(pool, cue)
            r.update(dataset=ds, detector=name, fa_budget=float(budget))
            rows.append(r)
        m = pool.score
        rows.append({"dataset": ds, "detector": "AUC_window_CNN_oof", "coverage_%": roc_auc_score(pool.fog[m], oof_p[m])})
        np.savez_compressed(os.path.join(C.OUT, "oof_cnn_%s.npz" % ds), cnn=oof_p,
                            **{k.replace("|", "__"): v for k, v in oof.items()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT, "cv_summary_cnn.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or C.DEV_DATASETS)
