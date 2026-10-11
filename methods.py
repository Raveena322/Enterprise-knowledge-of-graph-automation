"""Three methods compared in the paper: Rule-Based, Open-IE-style, and the Proposed
(word + character n-gram TF-IDF + engineered features, trained with SGD)."""
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from dataset import RELATIONS

# ---------------------------------------------------------------- 1) Rule-based
TRIGGERS = {
    "works_for": ["member of", "works in", "assigned to", "joined the"],
    "reports_to": ["reports to", "direct manager", "supervisor is", "report to"],
    "manages": ["manages", "overseen by", "responsible for managing"],
    "uses": ["relies on", "primary tool used by", "adopted"],
    "produces": ["product team behind", "developed and shipped by", "owns the roadmap"],
    "contributes_to": ["contributing engineering resources", "assigned to work on", "depends on deliverables"],
}


def rule_based_predict(sentence, subject, relation, obj):
    sent_l = sentence.lower()
    has_trigger = any(t in sent_l for t in TRIGGERS.get(relation, []))
    return int(has_trigger and subject.lower() in sent_l and obj.lower() in sent_l)


# ---------------------------------------------------------------- 2) Open-IE-style
GENERIC_VERB_CUES = ["is", "was", "are", "were", "reports", "manages", "relies",
                     "adopted", "owns", "joined", "assigned", "developed",
                     "depends", "member", "supervisor", "overseen", "contributing"]


def openie_predict(sentence, subject, relation, obj):
    sent_l = sentence.lower()
    subj_l, obj_l = subject.lower(), obj.lower()
    if not (subj_l in sent_l and obj_l in sent_l):
        return 0
    s_idx, o_idx = sent_l.find(subj_l), sent_l.find(obj_l)
    between = sent_l[min(s_idx, o_idx):max(s_idx, o_idx)]
    return int(any(v in between for v in GENERIC_VERB_CUES))


# ---------------------------------------------------------------- 3) Proposed
def make_feature_text(row):
    return row["sentence"]


def _numeric_features(rows):
    feats = np.zeros((len(rows), len(RELATIONS) + 4), dtype=np.float64)
    for i, r in enumerate(rows):
        sent_l = r["sentence"].lower()
        subj_l, obj_l = r["subject"].lower(), r["object"].lower()
        has_subj, has_obj = subj_l in sent_l, obj_l in sent_l
        trigger_count = sum(1 for t in TRIGGERS.get(r["relation"], []) if t in sent_l)
        if has_subj and has_obj:
            dist = abs(sent_l.find(subj_l) - sent_l.find(obj_l)) / max(len(sent_l), 1)
        else:
            dist = 1.0
        feats[i, 0], feats[i, 1], feats[i, 2], feats[i, 3] = float(has_subj), float(has_obj), trigger_count, dist
        feats[i, 4 + RELATIONS.index(r["relation"])] = 1.0
    return feats


class ProposedModel:
    def __init__(self, word_features=1500, char_features=1500):
        self.word_vec = TfidfVectorizer(max_features=word_features, ngram_range=(1, 2), sublinear_tf=True)
        self.char_vec = TfidfVectorizer(max_features=char_features, analyzer="char_wb",
                                        ngram_range=(3, 5), sublinear_tf=True)
        self.clf = SGDClassifier(loss="log_loss", random_state=0, alpha=3e-5, learning_rate="optimal")
        self.classes_ = np.array([0, 1])

    def _build_matrix(self, rows, fit=False):
        texts = [make_feature_text(r) for r in rows]
        if fit:
            Xw, Xc = self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)
        else:
            Xw, Xc = self.word_vec.transform(texts), self.char_vec.transform(texts)
        return sp.hstack([Xw, Xc, sp.csr_matrix(_numeric_features(rows))]).tocsr()

    def fit_transform_rows(self, rows):
        return self._build_matrix(rows, fit=True)

    def transform_rows(self, rows):
        return self._build_matrix(rows, fit=False)

    def train_epochs(self, X_train, y_train, X_val, y_val, n_epochs=10):
        hist = {"train_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": []}
        rng = np.random.RandomState(0)
        for _ in range(n_epochs):
            order = rng.permutation(X_train.shape[0])
            self.clf.partial_fit(X_train[order], y_train[order], classes=self.classes_)
            hist["train_loss"].append(log_loss(y_train, self.clf.predict_proba(X_train), labels=self.classes_))
            vp = self.clf.predict(X_val)
            hist["val_accuracy"].append(accuracy_score(y_val, vp))
            hist["val_precision"].append(precision_score(y_val, vp, zero_division=0))
            hist["val_recall"].append(recall_score(y_val, vp, zero_division=0))
        return hist

    def predict(self, X):
        return self.clf.predict(X)


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }


def measure_throughput(predict_fn, rows, repeats=3):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        for r in rows:
            predict_fn(r["sentence"], r["subject"], r["relation"], r["object"])
        rate = len(rows) / (time.perf_counter() - t0)
        best = rate if best is None else max(best, rate)
    return best


def measure_ml_throughput(model, rows, repeats=3):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(model.transform_rows(rows))
        rate = len(rows) / (time.perf_counter() - t0)
        best = rate if best is None else max(best, rate)
    return best
