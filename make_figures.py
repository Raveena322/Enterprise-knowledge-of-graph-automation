"""Generates the figures from results/results.json (run run_experiment.py first)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/results.json") as f:
    R = json.load(f)
OUT = "figures/"
os.makedirs(OUT, exist_ok=True)
ep = np.arange(1, 11)
H = R["proposed_training_history"]

# Fig 1: validation accuracy and training loss
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, np.array(H["val_accuracy"]) * 100, "o-", color="red", label="Proposed (trained)")
ax[0].plot(ep, R["rule_based_flat"], "s--", color="green", label="Rule-Based (no training)")
ax[0].plot(ep, R["openie_flat"], "^--", color="blue", label="Open-IE-style (no training)")
ax[0].set(title="Validation Accuracy vs. Epoch", xlabel="Epoch", ylabel="Accuracy (%)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].plot(ep, H["train_loss"], "o-", color="red")
ax[1].set(title="Proposed Model Training Loss vs. Epoch", xlabel="Epoch", ylabel="Log Loss"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + "fig1_accuracy_loss.png", dpi=150); plt.close()

# Fig 2: precision / recall per epoch
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, key, name in ((ax[0], "val_precision", "precision"), (ax[1], "val_recall", "recall")):
    a.plot(ep, np.array(H[key]) * 100, "o-", color="red", label="Proposed")
    a.plot(ep, [R["rule_based"][name]] * 10, "s--", color="green", label="Rule-Based")
    a.plot(ep, [R["openie"][name]] * 10, "^--", color="blue", label="Open IE")
    a.set(title=f"{name.capitalize()} vs. Epoch (Validation Set)", xlabel="Epoch", ylabel=f"{name.capitalize()} (%)")
    a.legend(fontsize=7); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + "fig2_precision_recall.png", dpi=150); plt.close()

# Fig 3: throughput (log scale)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
vals = [R["rule_based"]["throughput"], R["openie"]["throughput"], R["proposed"]["throughput"]]
bars = ax.bar(["Rule-Based", "Open IE", "Proposed"], vals, color=["green", "blue", "red"])
ax.set_yscale("log")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,.0f}", ha="center", fontweight="bold", fontsize=9)
ax.set(title="Measured Throughput (Triples/Second, Log Scale, Single CPU Core)", ylabel="Triples per second")
plt.tight_layout(); plt.savefig(OUT + "fig3_throughput.png", dpi=150); plt.close()

# Fig 4: confusion matrix
cm = np.array(R["proposed_confusion_matrix"])
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Valid", "Predicted Invalid"]); ax.set_yticklabels(["Actual Valid", "Actual Invalid"])
ax.set_title(f"Confusion Matrix - Proposed ({R['dataset_sizes']['test']} Test Triples)")
plt.colorbar(im, fraction=.046, pad=.04); plt.tight_layout(); plt.savefig(OUT + "fig4_confusion_matrix.png", dpi=150); plt.close()

# Fig 5: F1 under perturbed conditions
order = ["structured", "semi_structured", "unstructured", "noisy_ocr", "multilingual"]
labels = ["Structured\n(clean)", "Semi-\nStructured", "Unstructured", "Noisy/OCR", "Multilingual-like"]
x = np.arange(5); w = .25
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for off, key, name, col in ((-w, "proposed_f1", "Proposed", "red"), (0, "rule_based_f1", "Rule-Based", "green"), (w, "openie_f1", "Open IE", "blue")):
    ax.bar(x + off, [R["condition_results"][c][key] for c in order], w, label=name, color=col)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("F1-Score (%)")
ax.set_title("F1-Score Under Perturbed Document Conditions"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig5_f1_conditions.png", dpi=150); plt.close()
print("figures written")
