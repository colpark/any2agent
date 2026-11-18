"""TH-4 figures: (1) teacher BA vs head BA per class (Astra-taught on Test200, Qwen-taught on Test); (2) label-efficiency curves."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from th_common import TH, CLASSES
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Astra (Test200)": "#2a78d6", "Qwen3.5-9B (Test)": "#eb6834"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
E1, E2 = json.load(open(f"{TH}/e1.json")), json.load(open(f"{TH}/e2.json")); os.makedirs(f"{TH}/figs", exist_ok=True)

fig, ax = plt.subplots(figsize=(5.2, 5)); ax.plot([0.45, 1], [0.45, 1], color=INK2, lw=1, ls="--", zorder=1)
for lab, src, key in [("Astra (Test200)", "astra+code", "astra_test200"), ("Qwen3.5-9B (Test)", "qwen", "qwen_test")]:
    pts = [(E1["grades"][c][key]["ba"], r["heads"][src]["ba"], c) for c, r in E2["per_class"].items()
           if src in r["heads"] and "diff" in r["heads"][src] and E1["grades"][c].get(key) and E1["grades"][c][key]["ba"] is not None]
    if src == "astra+code":  # head BA on the same Test200 patches as the teacher: teacher BA + paired diff
        pts = [(t, t + E2["per_class"][c]["heads"][src]["diff"], c) for t, _, c in pts]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=40, color=COL[lab], edgecolor=SURF, linewidth=2, label=lab, zorder=3)
    for t, h, c in pts:
        if abs(h - t) > 0.08: ax.annotate(CLASSES[int(c) - 1][:18], (t, h), xytext=(4, -10), textcoords="offset points", fontsize=7, color=INK2)
ax.set_xlabel("Teacher balanced accuracy"); ax.set_ylabel("Head balanced accuracy (same patches)"); ax.set_xlim(0.45, 1); ax.set_ylim(0.45, 1)
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper left"); ax.set_title("Heads vs their teachers, per class (dashed: equal)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{TH}/figs/e2_teacher_vs_head.png", dpi=150)

cls = list(E2["per_class"]); n = len(cls); cols = 5; rows = -(-n // cols)
fig, axs = plt.subplots(rows, cols, figsize=(12, 2.3 * rows), sharex=True, sharey=True)
for ax, c in zip(axs.ravel(), cls):
    r = E2["per_class"][c]; ns = sorted(int(k) for k in r["ref_curve"]); ys = [r["ref_curve"][str(k)] for k in ns]
    ax.plot(ns, ys, color=INK2, lw=2, marker="o", ms=4, label="Head on n true labels (mean of 3 seeds)")
    if "astra+code" in r["heads"]: ax.axhline(r["heads"]["astra+code"]["ba"], color=COL["Astra (Test200)"], lw=2, label="Astra + code head (500 teacher labels, 0 true)")
    ax.set_xscale("log"); ax.set_title(f"{c}. {CLASSES[int(c) - 1][:24]}", fontsize=8, loc="left"); ax.grid(color=GRID, lw=0.5); ax.set_ylim(0.45, 1)
for ax in axs.ravel()[n:]: ax.axis("off")
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=2, loc="upper center", fontsize=9); fig.supxlabel("True labels (log n)", color=INK2); fig.supylabel("Balanced accuracy on Test", color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{TH}/figs/e2_label_efficiency.png", dpi=150); print("figs written")
