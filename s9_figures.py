import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fi_config as C

SHORT = {"D0_FI": "D0 FI", "D1_FI_gate": "D1 +gate", "D2_FI_gate_smooth": "D2 +smooth",
         "D3_FI_gate_smooth_hyst": "D3 +hyst", "D4_D3_plus_arrest": "D4 +arrest", "D5_LR_hyst": "D5 LR",
         "D6_CNN_hyst": "D6 CNN"}


def read(name):
    p = os.path.join(C.OUT, name)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def bars(df, value, title, fname, ylabel):
    if df.empty:
        return
    piv = df.pivot_table(index="detector", columns="dataset", values=value)
    piv = piv.loc[[d for d in SHORT if d in piv.index]]
    piv.index = [SHORT[d] for d in piv.index]
    ax = piv.plot.bar(figsize=(8, 3.6), width=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, fname), dpi=200)
    plt.close()


def main():
    dev = pd.concat([read("cv_summary.csv"), read("cv_summary_cnn.csv")], ignore_index=True)
    if not dev.empty:
        d = dev[(dev["fa_budget"] == C.MAIN_BUDGET) & dev["detector"].isin(list(SHORT))]
        d.to_csv(os.path.join(C.OUT, "table_development.csv"), index=False)
        bars(d, "coverage_%", "Development (subject-grouped CV): episode coverage", "fig_dev_coverage.png", "Coverage (%)")
        bars(d, "fa_per_h", "Development (subject-grouped CV): false alarms per hour", "fig_dev_fa.png", "False alarms / h")
        bars(d, "timely_%", "Development (subject-grouped CV): timely activation", "fig_dev_timely.png", "Timely (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        for ds, g in dev[dev["detector"].isin(list(SHORT))].groupby("dataset"):
            for det, h in g.groupby("detector"):
                h = h.sort_values("fa_budget")
                ax.plot(h["fa_per_h"], h["coverage_%"], marker="o", label="%s %s" % (ds, SHORT[det]))
        ax.set_xlabel("False alarms / h")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Coverage against false alarms (budgets 5, 10, 15 / h)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(color="0.9")
        plt.tight_layout()
        plt.savefig(os.path.join(C.OUT, "fig_dev_tradeoff.png"), dpi=200)
        plt.close()
    ext = read("external_chance.csv")
    if not ext.empty:
        ext.to_csv(os.path.join(C.OUT, "table_external.csv"), index=False)
        bars(ext, "coverage_%", "External test (frozen): episode coverage", "fig_ext_coverage.png", "Coverage (%)")
        bars(ext, "fa_per_h", "External test (frozen): false alarms per hour", "fig_ext_fa.png", "False alarms / h")
    kin = read("failure_by_fog_type.csv")
    if not kin.empty:
        k = kin[kin["split"] == "kinetic"]
        for ds, g in k.groupby("dataset"):
            piv = g.pivot(index="detector", columns="category", values="coverage_%")
            piv = piv.loc[[x for x in SHORT if x in piv.index]]
            piv.index = [SHORT[x] for x in piv.index]
            ax = piv.plot.bar(figsize=(7, 3.4), width=0.8)
            ax.set_ylabel("Coverage (%)")
            ax.set_title("%s: kinetic vs akinetic episodes" % ds)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(C.OUT, "fig_kinetic_%s.png" % ds), dpi=200)
            plt.close()
    sev = read("external_fogstar_by_severity.csv")
    if not sev.empty:
        piv = sev.pivot(index="detector", columns="severity", values="coverage_%")
        piv = piv.loc[[x for x in SHORT if x in piv.index]]
        piv.index = [SHORT[x] for x in piv.index]
        ax = piv.plot.bar(figsize=(7, 3.4), width=0.8)
        ax.set_ylabel("Coverage (%)")
        ax.set_title("FoG-STAR (external): coverage by FoG type")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(C.OUT, "fig_ext_fogstar_types.png"), dpi=200)
        plt.close()
    print("figures and tables written to", C.OUT)
    for f in sorted(os.listdir(C.OUT)):
        if f.startswith(("fig_", "table_")):
            print("  ", f)


if __name__ == "__main__":
    main()
