"""Metaheuristic feature-weight extractive summariser.

score(s) = w . x(s), with x(s) the 13 document-normalised features of sentence s and w in [0, 1]^13 the
weight vector found by a metaheuristic. The top-k sentences (k = ceil(ratio * n)) are returned in their
original order. During training k equals the number of reference sentences of each document, and the
fitness is the mean of ROUGE-1 F1 and ROUGE-2 F1 over the training documents (maximised).
"""
import hashlib
import json
import math
from importlib import resources
from pathlib import Path

import numpy as np

from ..features import FEATURE_NAMES, SentenceFeatureExtractor, split_sentences
from ..metrics import FastRouge, evaluate_summary

PRETRAINED_DIR = "pretrained_optimal_models"
FORMAT_VERSION = 1


def _pretrained_root():
    return resources.files("satuasummarization.extractive.metaheuristics").joinpath(PRETRAINED_DIR)


def content_sha256(model_dict):
    """sha256 of the canonical JSON of a model (independent of line endings and key order of the file)."""
    canonical = json.dumps(model_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def available_models():
    """Index of the bundled pretrained models (algorithm -> metadata incl. test metrics)."""
    return json.loads(_pretrained_root().joinpath("index.json").read_text(encoding="utf-8"))


def _top_k_mask(scores, k):
    k = int(min(max(k, 1), len(scores)))
    mask = np.zeros(len(scores), bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


class MetaheuristicSummarizer:
    def __init__(self, weights=None, algorithm=None, ratio=0.3, extractor=None, meta=None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"weights must have {len(FEATURE_NAMES)} values")
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.algorithm, self.ratio = algorithm, ratio
        self._extractor = extractor
        self.meta = meta or {}

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_pretrained(cls, name="best", extractor=None, ratio=None):
        """Load a bundled model by algorithm acronym (e.g. 'PLO') or 'best' (highest test ROUGE)."""
        index = available_models()
        key = index["best"] if name == "best" else name
        if key not in index["models"]:
            raise ValueError(f"unknown pretrained model {name!r}; available: {sorted(index['models'])}")
        entry = index["models"][key]
        model = json.loads(_pretrained_root().joinpath(entry["file"]).read_text(encoding="utf-8"))
        if content_sha256(model) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for pretrained model {key}")
        return cls.from_dict(model, extractor=extractor, ratio=ratio)

    @classmethod
    def load(cls, path, extractor=None, ratio=None):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), extractor=extractor, ratio=ratio)

    @classmethod
    def from_dict(cls, d, extractor=None, ratio=None):
        if d.get("format_version") != FORMAT_VERSION or d.get("feature_names") != FEATURE_NAMES:
            raise ValueError("model file was created for a different feature set or format")
        return cls(d["weights"], d["algorithm"], ratio or d.get("ratio", 0.3), extractor, d.get("meta", {}))

    def to_dict(self):
        self._check_fitted()
        return {"format_version": FORMAT_VERSION, "algorithm": self.algorithm, "ratio": self.ratio,
                "feature_names": FEATURE_NAMES, "weights": [round(float(w), 10) for w in self.weights],
                "meta": self.meta}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
        return path

    # ------------------------------------------------------------------ training
    def fit(self, docs, algorithm="GWO", n_agents=30, max_iter=100, seed=None, verbose=False, **params):
        """Optimise the weights on training documents (each needs ``features``, ``labels``, ``sentences``)."""
        from metaheuristicpy.core import get_optimizer

        if not docs:
            raise ValueError("docs must not be empty")
        feats = [np.asarray(d.features) for d in docs]
        ks = [int(np.asarray(d.labels).sum()) for d in docs]
        rouge = [FastRouge(d.sentences, d.labels) for d in docs]

        def fitness(W):
            total = np.zeros(len(W))
            for F, k, R in zip(feats, ks, rouge):
                S = F @ W.T                                         # (n_sentences, n_agents)
                top = np.argsort(-S, axis=0, kind="stable")[:k]
                masks = np.zeros((len(W), len(F)), bool)
                masks[np.arange(len(W))[None, :].repeat(k, 0), top] = True
                r1, r2 = R.f1(masks)
                total += (r1 + r2) / 2
            return total / len(docs)

        opt = get_optimizer(algorithm, n_agents=n_agents, max_iter=max_iter, seed=seed, verbose=verbose, **params)
        res = opt.optimize(fitness, len(FEATURE_NAMES), 0.0, 1.0, minimize=False, batch=True)
        self.weights, self.algorithm = res.best_position, algorithm
        self.meta.update(train_fitness=float(res.best_fitness), history=[float(h) for h in res.history],
                         n_iter=res.n_iter, fes=res.fes, train_time_s=round(res.time, 2),
                         n_agents=n_agents, max_iter=max_iter, seed=seed)
        return self

    # ------------------------------------------------------------------ inference
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = SentenceFeatureExtractor()
        return self._extractor

    def _check_fitted(self):
        if self.weights is None:
            raise RuntimeError("model has no weights: call fit() or from_pretrained()")

    def select(self, features, k=None, ratio=None):
        """Boolean mask of the selected sentences for a feature matrix."""
        self._check_fitted()
        F = np.asarray(features)
        k = k if k is not None else math.ceil((ratio or self.ratio) * len(F))
        return _top_k_mask(F @ self.weights, k)

    def summarize(self, text=None, title="", sentences=None, ratio=None):
        """Summarise raw text (split into sentences) or a list of sentences.

        Returns {'summary', 'sentences': [{'index', 'text', 'score', 'selected'}], 'algorithm', 'ratio'}."""
        self._check_fitted()
        if sentences is None:
            if not text or not str(text).strip():
                raise ValueError("provide text or sentences")
            sentences = split_sentences(text)
        if not sentences:
            raise ValueError("no sentence with words was found")
        ratio = ratio or self.ratio
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        F = self.extractor.transform(sentences, title)
        scores = F @ self.weights
        mask = _top_k_mask(scores, math.ceil(ratio * len(sentences)))
        return {
            "summary": " ".join(s for s, m in zip(sentences, mask) if m),
            "sentences": [{"index": i, "text": s, "score": round(float(sc), 6), "selected": bool(m)}
                          for i, (s, sc, m) in enumerate(zip(sentences, scores, mask))],
            "algorithm": self.algorithm, "ratio": ratio,
        }

    def evaluate(self, docs, length="oracle"):
        """Mean metrics over documents. length='oracle' uses k = #reference sentences, 'ratio' uses self.ratio."""
        rows = []
        for d in docs:
            k = int(np.sum(d.labels)) if length == "oracle" else None
            mask = self.select(d.features, k=k)
            rows.append(evaluate_summary(d.sentences, d.labels, mask))
        return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
