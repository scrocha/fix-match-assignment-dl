import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- Configuration ---------------------------------
MODEL_HINTS = [
    "ResNet34",
    "ResNet18_pretrained",
    "FixMatch",
]

# ---------------------------- Utilities -------------------------------------
def _infer_model_from_path(path: Path) -> Optional[str]:
    m = re.match(r"^FixMatch_\d+_labels_per_class(?:_(.+))?$", path.name, re.IGNORECASE)
    if m:
        bb = m.group(1)
        if bb:
            return f"fixmatch_{bb.lower()}"
        return "fixmatch"

def _infer_lpc_from_path(path: Path) -> Optional[int]:
    m = re.match(r"^FixMatch_(\d+)_labels_per_class(?:_.*)?$", path.name, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return val

@dataclass
class Record:
    source: Path
    model: Optional[str]
    lpc: Optional[int]
    df: pd.DataFrame


def scan_experiments(root: Path) -> List[Record]:
    records: List[Record] = []
    root = root.resolve()

    dirs = [
        p for p in root.rglob("FixMatch_*_labels_per_class*")
        if p.is_dir()
    ]

    for d in dirs:
        csv_file = d / "training_logs.csv"
        if not csv_file.exists():
            continue
        try:
            raw = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            continue
        if raw.empty:
            continue
        df = raw.copy()
        if "epoch" not in df.columns:
            df.insert(0, "epoch", np.arange(len(df), dtype=int))
        model = _infer_model_from_path(d)
        lpc = _infer_lpc_from_path(d)
        df["__source__"] = str(csv_file)
        records.append(Record(source=csv_file, model=model, lpc=lpc, df=df))

    return records

def build_master_df(records: List[Record]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for rec in records:
        df = rec.df.copy()
        df["model"] = rec.model
        df["lpc"] = rec.lpc

        if "loss" not in df.columns:
            if "loss_x" in df.columns or "loss_u" in df.columns:
                df["loss"] = df.get("loss_x", 0) + df.get("loss_u", 0)

        df["is_hybrid_loss"] = True
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    master = pd.concat(rows, ignore_index=True)

    for c in ["epoch", "lpc"]:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")

    if "test_acc" in master.columns:
        try:
            if master["test_acc"].dropna().between(0, 1).mean() > 0.8:
                master["test_acc"] = master["test_acc"] * 100.0
        except Exception:
            pass

    return master

# ---------------------------- Plotting & Analysis ---------------------------

def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def plot_loss_curves_by_lpc(master: pd.DataFrame, outdir: Path) -> List[Path]:
    paths: List[Path] = []

    for lpc in sorted([int(x) for x in master["lpc"].dropna().unique()]):
        sub = master[(master["lpc"] == lpc)]

        plt.figure(figsize=(8, 5))
        for model, dfm in sub.groupby("model"):
            g = dfm.sort_values("epoch")   
            plt.plot(g["epoch"], g["test_loss"], label=str(model))
        plt.xlabel("Época")
        plt.ylabel("Test loss")
        plt.title(f"Curvas de teste — {int(lpc)} rótulos/classe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = outdir / f"loss_curves_by_lpc_{int(lpc)}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()
        paths.append(fname)

    return paths


def plot_final_acc_vs_lpc(master: pd.DataFrame, outdir: Path) -> Optional[Path]:
    best_rows = []
    for (model, lpc, src), dfm in master.groupby(["model", "lpc", "__source__"]):
        if dfm["test_acc"].notna().any():
            idxmax = dfm["test_acc"].idxmax()
            if pd.notna(idxmax):
                row = dfm.loc[idxmax]
                best_rows.append({"model": model, "lpc": lpc, "src": src, "acc": float(row["test_acc"])})

    finals = pd.DataFrame(best_rows)

    plt.figure(figsize=(8,5))
    for model, dfg in finals.groupby("model"):
        dfg = dfg.sort_values("lpc")
        plt.errorbar(dfg["lpc"], dfg["acc"], marker="o", label=str(model))

    plt.xscale("log")
    plt.xticks([1, 4, 25, 250, 400], labels=["1", "4", "25", "250", "400"])
    plt.xlabel("Rótulos por classe (escala log)")
    plt.ylabel("Melhor test acc (%)")
    plt.title("Melhor test accuracy vs. quantidade de dados rotulados")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fname = outdir / "final_acc_vs_lpc.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    return fname

def plot_hybrid_by_epoch_losses(master: pd.DataFrame, outdir: Path) -> List[Path]:
    made: List[Path] = []
    tmp = master.copy()

    if "loss" in tmp.columns:
        tmp["total_loss_plot"] = pd.to_numeric(tmp["loss"], errors="coerce")

    # Apenas LPC 1 e 400
    target_lpcs = [1, 400]
    for lpc in target_lpcs:
        dfl = tmp[tmp["lpc"] == lpc]
        if dfl.empty:
            continue

        plt.figure(figsize=(9,5))
        for model, dfm in dfl.groupby("model"):
            dfm = dfm.sort_values("epoch")
            agg = dfm.groupby("epoch").agg(unsup=("loss_u", "mean")).reset_index()
            plt.plot(agg["epoch"], agg["unsup"], label=f"{model}")
        plt.xlabel("Época")
        plt.ylabel("loss_u")
        plt.title(f"loss_u por época — {int(lpc)} rótulos/classe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname_u = outdir / f"hybrid_unsup_loss_by_epoch_{int(lpc)}.png"
        plt.tight_layout(); plt.savefig(fname_u, dpi=160); plt.close()
        made.append(fname_u)

        plt.figure(figsize=(9,5))
        for model, dfm in dfl.groupby("model"):
            dfm = dfm.sort_values("epoch")
            agg = dfm.groupby("epoch").agg(sup=("loss_x", "mean")).reset_index()
            plt.plot(agg["epoch"], agg["sup"], label=f"{model}")
        plt.xlabel("Época")
        plt.ylabel("loss_x")
        plt.title(f"loss_x por época — {int(lpc)} rótulos/classe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname_x = outdir / f"hybrid_sup_loss_by_epoch_{int(lpc)}.png"
        plt.tight_layout(); plt.savefig(fname_x, dpi=160); plt.close()
        made.append(fname_x)

        plt.figure(figsize=(9,5))
        for model, dfm in dfl.groupby("model"):
            dfm = dfm.sort_values("epoch")
            agg = dfm.groupby("epoch").agg(total=("total_loss_plot", "mean")).reset_index()
            plt.plot(agg["epoch"], agg["total"], label=f"{model}")
        plt.xlabel("Época")
        plt.ylabel("loss (total)")
        plt.title(f"Perda total por época — {int(lpc)} rótulos/classe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname_t = outdir / f"hybrid_total_loss_by_epoch_{int(lpc)}.png"
        plt.tight_layout(); plt.savefig(fname_t, dpi=160); plt.close()
        made.append(fname_t)

    return made

def plot_hybrid_by_epoch_accuracy(master: pd.DataFrame, outdir: Path) -> List[Path]:
    made: List[Path] = []
    tmp = master.copy() 

    for lpc, dfl in tmp.groupby("lpc"):
        plt.figure(figsize=(9,5))
        for model, dfm in dfl.groupby("model"):
            agg = dfm.sort_values("epoch")
            plt.plot(agg["epoch"], agg["test_acc"], label=str(model))
        plt.xlabel("Época")
        plt.ylabel("Test accuracy (%)")
        plt.title(f"Acurácia de teste por época — {int(lpc)} rótulos/classe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = outdir / f"hybrid_accuracy_by_epoch_{int(lpc)}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()
        made.append(fname)

    return made

# ---------------------------- Main ------------------------------------------
def main():
    root = Path("experiments")
    outdir = root / "analysis"
    _ensure_outdir(outdir)

    print(f"Scanning {root} ...")
    records = scan_experiments(root)

    print(f"Construindo dataframe mestre a partir de {len(records)} execuções ...")
    master = build_master_df(records)
    master.to_csv(outdir / "master_metrics.csv", index=False)

    print("Gerando gráficos ...")
    plot_loss_curves_by_lpc(master, outdir)
    plot_final_acc_vs_lpc(master, outdir)
    plot_hybrid_by_epoch_losses(master, outdir)
    plot_hybrid_by_epoch_accuracy(master, outdir)

    print(f"Resultados salvos em: {outdir}")

if __name__ == "__main__":
    main()