"""Clean latency figure: decode latency vs the 100 ms EEG window deadline.

One claim: inference latency is not the bottleneck. Log-scale bars sit ~1500x
under the deadline line. Reproducible from results/latency_benchmark.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "paper-kit"))
import paperstyle  # noqa: E402

paperstyle.use()
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEADLINE_MS = 100.0  # bench/run.py: 2.0 s window, 100 ms per-window deadline
ROOT = Path(__file__).resolve().parents[1]
P = paperstyle.PALETTE

LABELS = {
    ("RandomForest_200t", "stream"): "RF\nstream",
    ("RandomForest_200t", "batch_64"): "RF\nbatch64",
    ("MLP_256_128_64", "stream"): "MLP\nstream",
    ("MLP_256_128_64", "batch_64"): "MLP\nbatch64",
}
BAR = paperstyle.PALETTE["blue"]  # same method-family blue across figures
TAIL = "#aebfd0"  # p99 whisker shade


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "latency_benchmark.csv")
    df = df[[(r.model, r.mode) in LABELS for r in df.itertuples()]]
    order = list(LABELS)
    df["key"] = list(zip(df["model"], df["mode"]))
    df = df.set_index("key").loc[order].reset_index()

    names = [LABELS[k] for k in order]
    p50 = df["p50_ms"].to_numpy()
    p99 = df["p99_ms"].to_numpy()
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    # single bar per config = p99 (conservative worst case)
    ax.bar(x, p99, width=0.6, color=BAR)

    ax.set_yscale("log")
    ax.set_ylim(0.002, 400)
    ax.axhline(DEADLINE_MS, ls="--", lw=1.2, color="#b3402f")
    ax.text(-0.45, DEADLINE_MS * 1.3, "100 ms EEG window deadline",
            ha="left", va="bottom", fontsize=8, color="#b3402f")

    ax.set_ylabel("p99 decode latency (ms, log)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.5)
    for xi, v in zip(x, p99):
        ax.text(xi, v * 1.4, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    # headroom annotation on the tallest bar
    ax.annotate("", xy=(3, DEADLINE_MS), xytext=(3, p99[3]),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8))
    ax.text(2.62, 1.5, "$\\sim$1500$\\times$\nheadroom", fontsize=7.5,
            color="#555", va="center", ha="right")

    out = ROOT / "docs" / "paper" / "figs" / "latency.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
