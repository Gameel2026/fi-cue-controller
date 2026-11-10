import os
import numpy as np
import pandas as pd
import fi_config as C

LEVELS = ["any", "25", "50"]


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main():
    path = os.path.join(C.OUT, "posthoc", "strict_coverage.csv")
    a = pd.read_csv(path)
    a = a[a["level"].isin(LEVELS)].copy()
    a["q_bh_stratum"] = np.nan
    a["q_bh_both_strata"] = np.nan
    for det, g in a.groupby("detector"):
        a.loc[g.index, "q_bh_both_strata"] = bh(g["p_one_sided"].values)
        for st, h in g.groupby("stratum"):
            a.loc[h.index, "q_bh_stratum"] = bh(h["p_one_sided"].values)
    out = os.path.join(C.OUT, "posthoc", "strict_coverage_bh.csv")
    a.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["dataset", "detector", "stratum", "level", "observed_pct", "chance_pct", "gain_points",
            "p_one_sided", "q_bh_stratum", "q_bh_both_strata"]
    print(a[cols].round(3).to_string(index=False))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
