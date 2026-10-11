"""Runs the full Phase-I experiment and writes results.json."""
import json
import os
import random
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from dataset import build_dataset, _typo
from methods import (rule_based_predict, openie_predict, ProposedModel,
                     evaluate_predictions, measure_throughput, measure_ml_throughput)

# Seed after importing project modules so imports cannot overwrite this seed.
random.seed(7)
R = {}

# 1. Dataset, 80/10/10 split
data = build_dataset(n_per_relation=260)
train, temp = train_test_split(data, test_size=0.2, random_state=0, stratify=[r["label"] for r in data])
val, test = train_test_split(temp, test_size=0.5, random_state=0, stratify=[r["label"] for r in temp])
R["dataset_sizes"] = {"train": len(train), "val": len(val), "test": len(test), "total": len(data)}
y_val = np.array([r["label"] for r in val])
y_test = np.array([r["label"] for r in test])

# 2-3. Baselines (no training)
for name, fn in (("rule_based", rule_based_predict), ("openie", openie_predict)):
    preds = [fn(r["sentence"], r["subject"], r["relation"], r["object"]) for r in test]
    R[name] = evaluate_predictions(y_test, preds)
    R[name]["throughput"] = measure_throughput(fn, test)

# 4. Proposed model, noise-augmented training, 10 epochs
aug = list(train)
for r in train:
    if random.random() < 0.35:
        pr = dict(r)
        pr["sentence"] = _typo(r["sentence"], rate=0.1)
        aug.append(pr)
random.shuffle(aug)
y_train = np.array([r["label"] for r in aug])

model = ProposedModel()
X_train = model.fit_transform_rows(aug)
X_val, X_test = model.transform_rows(val), model.transform_rows(test)
R["proposed_training_history"] = model.train_epochs(X_train, y_train, X_val, y_val, n_epochs=10)
test_preds = model.predict(X_test)
R["proposed"] = evaluate_predictions(y_test, test_preds)
R["proposed"]["throughput"] = measure_ml_throughput(model, test)
R["proposed_confusion_matrix"] = confusion_matrix(y_test, test_preds, labels=[1, 0]).tolist()
R["rule_based_flat"] = [R["rule_based"]["accuracy"]] * 10
R["openie_flat"] = [R["openie"]["accuracy"]] * 10

# 5. Robustness under programmatic perturbations
FOREIGN = ["dass", "que", "desu", "ki", "et"]


def p_ocr(t): return _typo(t, rate=0.12)
def p_unstructured(t): return re.sub(r"[.,]", "", t).lower()
def p_semi(t): return t.replace(" is ", ": ").replace(" was ", ": ").rstrip(".")


def p_multi(t):
    tok = t.split()
    if len(tok) > 4:
        tok.insert(len(tok) // 2, random.choice(FOREIGN))
    return " ".join(tok)


conditions = {"structured": lambda t: t, "semi_structured": p_semi, "unstructured": p_unstructured,
              "noisy_ocr": p_ocr, "multilingual": p_multi}
R["condition_results"] = {}
for cname, fn in conditions.items():
    pt = [dict(r, sentence=fn(r["sentence"])) for r in test]
    y_c = np.array([r["label"] for r in pt])
    rb = [rule_based_predict(r["sentence"], r["subject"], r["relation"], r["object"]) for r in pt]
    oi = [openie_predict(r["sentence"], r["subject"], r["relation"], r["object"]) for r in pt]
    pp = model.predict(model.transform_rows(pt))
    R["condition_results"][cname] = {
        "rule_based_f1": evaluate_predictions(y_c, rb)["f1"],
        "openie_f1": evaluate_predictions(y_c, oi)["f1"],
        "proposed_f1": evaluate_predictions(y_c, pp)["f1"],
    }

os.makedirs("results", exist_ok=True)
with open("results/results.json", "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps({k: R[k] for k in ("dataset_sizes", "rule_based", "openie", "proposed",
                                     "proposed_confusion_matrix", "condition_results")}, indent=2))
