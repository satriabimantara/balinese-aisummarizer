"""Supervised extractive summariser: a binary sentence classifier decides which sentences form the summary.

P(selected | x(s)) is predicted from the 13 document-normalised features of sentence s. By default the summary
holds every sentence with P >= ``threshold`` (at least one sentence is always returned); with ``ratio`` (or
``k``) the ceil(ratio * n) most probable sentences are taken instead, like ``MetaheuristicSummarizer``.
"""
import hashlib
import io
import json
import math
from importlib import resources
from pathlib import Path

import numpy as np

from ..features import FEATURE_NAMES, SentenceFeatureExtractor, split_sentences
from ..metaheuristics.summarizer import _top_k_mask
from ..metrics import evaluate_summary
from .models import TORCH_MODELS, TorchSentenceClassifier, build_model

PRETRAINED_DIR = "pretrained_models"
FORMAT_VERSION = 1


def _pretrained_root():
    return resources.files("satuasummarization.extractive.supervisedlearning").joinpath(PRETRAINED_DIR)


def available_supervised_models():
    """Index of the bundled pretrained classifiers (name -> metadata incl. cross-validation and test metrics)."""
    return json.loads(_pretrained_root().joinpath("index.json").read_text(encoding="utf-8"))


class SupervisedSummarizer:
    def __init__(self, model=None, name=None, ratio=None, threshold=0.5, extractor=None, meta=None):
        if ratio is not None and not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.model, self.name, self.ratio, self.threshold = model, name, ratio, threshold
        self._extractor = extractor
        self.meta = meta or {}

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_pretrained(cls, name="best", extractor=None, ratio=None, threshold=0.5):
        """Load a bundled model by name (e.g. 'RandomForest') or 'best' (highest cross-validation macro F1)."""
        index = available_supervised_models()
        key = index["best"] if name == "best" else name
        if key not in index["models"]:
            raise ValueError(f"unknown pretrained model {name!r}; available: {sorted(index['models'])}")
        entry = index["models"][key]
        raw = _pretrained_root().joinpath(entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch for pretrained model {key}")
        return cls._from_bytes(raw, entry["file"], extractor, ratio, threshold)

    @classmethod
    def load(cls, path, extractor=None, ratio=None, threshold=0.5):
        """Load a model written by ``save``. A .joblib file is a pickle: only load files you trust."""
        return cls._from_bytes(Path(path).read_bytes(), str(path), extractor, ratio, threshold)

    @classmethod
    def _from_bytes(cls, raw, filename, extractor, ratio, threshold):
        if filename.endswith(".npz"):
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                header = json.loads(str(z["__header__"]))
                arrays = {k: z[k] for k in z.files if k != "__header__"}
            model = TorchSentenceClassifier.from_state(header["config"], arrays)
        else:
            import joblib
            import sklearn

            payload = joblib.load(io.BytesIO(raw))
            header, model = payload["header"], payload["model"]
            if header.get("sklearn_version") != sklearn.__version__:
                raise RuntimeError(f"model was saved with scikit-learn {header.get('sklearn_version')}, but "
                                   f"{sklearn.__version__} is installed; install the same version or retrain")
        if header.get("format_version") != FORMAT_VERSION or header.get("feature_names") != FEATURE_NAMES:
            raise ValueError("model file was created for a different feature set or format")
        return cls(model, header["name"], ratio, threshold, extractor, header.get("meta", {}))

    def save(self, path):
        """Write the model: PyTorch models as .npz (arrays only), scikit-learn models as .joblib (pickle)."""
        self._check_fitted()
        header = {"format_version": FORMAT_VERSION, "name": self.name, "feature_names": FEATURE_NAMES, "meta": self.meta}
        path = Path(path)
        if isinstance(self.model, TorchSentenceClassifier):
            header["config"], arrays = self.model.state()
            path = path.with_suffix(".npz")
            np.savez_compressed(path, __header__=np.array(json.dumps(header)), **arrays)
        else:
            import joblib
            import sklearn

            header["sklearn_version"] = sklearn.__version__
            path = path.with_suffix(".joblib")
            joblib.dump({"header": header, "model": self.model}, path, compress=3)
        return path

    # ------------------------------------------------------------------ training
    def fit(self, docs, model="RandomForest", seed=42):
        """Train a classifier on documents that carry ``features`` and ``labels`` (see ``models.MODEL_INFO``)."""
        if not docs:
            raise ValueError("docs must not be empty")
        X = np.vstack([np.asarray(d.features) for d in docs])
        y = np.concatenate([np.asarray(d.labels) for d in docs])
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have {len(FEATURE_NAMES)} columns")
        groups = np.repeat(np.arange(len(docs)), [len(d.labels) for d in docs])
        self.model, self.name = build_model(model, seed).fit(X, y, groups), model
        self.meta.update(seed=seed, n_train_docs=len(docs), n_train_sentences=int(len(y)))
        return self

    # ------------------------------------------------------------------ inference
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = SentenceFeatureExtractor()
        return self._extractor

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("model is not trained: call fit() or from_pretrained()")

    def predict_proba(self, features):
        """P(selected) of every sentence of ONE document (rows of ``features``)."""
        self._check_fitted()
        return self.model.predict_proba(np.asarray(features, dtype=float))[:, 1]

    def _mask(self, proba, k=None, ratio=None, threshold=None):
        ratio = ratio or self.ratio
        if k is None and ratio is not None:
            if not 0 < ratio <= 1:
                raise ValueError("ratio must be in (0, 1]")
            k = math.ceil(ratio * len(proba))
        if k is not None:
            return _top_k_mask(proba, k)
        mask = proba >= (threshold or self.threshold)
        return mask if mask.any() else _top_k_mask(proba, 1)

    def select(self, features, k=None, ratio=None, threshold=None):
        """Boolean mask of the selected sentences of one document."""
        return self._mask(self.predict_proba(features), k, ratio, threshold)

    def summarize(self, text=None, title="", sentences=None, ratio=None, threshold=None):
        """Summarise raw text (split into sentences) or a list of sentences.

        Returns {'summary', 'sentences': [{'index', 'text', 'score', 'selected'}], 'algorithm', 'ratio',
        'threshold'}; 'score' is P(selected) and 'ratio' is None when the threshold decided the length."""
        self._check_fitted()
        if sentences is None:
            if not text or not str(text).strip():
                raise ValueError("provide text or sentences")
            sentences = split_sentences(text)
        if not sentences:
            raise ValueError("no sentence with words was found")
        proba = self.predict_proba(self.extractor.transform(sentences, title))
        mask = self._mask(proba, None, ratio, threshold)
        ratio = ratio or self.ratio
        return {
            "summary": " ".join(s for s, m in zip(sentences, mask) if m),
            "sentences": [{"index": i, "text": s, "score": round(float(p), 6), "selected": bool(m)}
                          for i, (s, p, m) in enumerate(zip(sentences, proba, mask))],
            "algorithm": self.name, "ratio": ratio, "threshold": None if ratio else (threshold or self.threshold),
        }

    def evaluate(self, docs, length="threshold"):
        """Mean metrics over documents. length='threshold' keeps P >= threshold, 'oracle' uses k = #reference
        sentences, 'ratio' uses self.ratio."""
        if length not in ("threshold", "oracle", "ratio"):
            raise ValueError("length must be 'threshold', 'oracle' or 'ratio'")
        if length == "ratio" and self.ratio is None:
            raise ValueError("set ratio to evaluate with length='ratio'")
        rows = []
        for d in docs:
            proba = self.predict_proba(d.features)
            if length == "threshold":
                mask = proba >= self.threshold
                mask = mask if mask.any() else _top_k_mask(proba, 1)
            else:
                mask = self._mask(proba, int(np.sum(d.labels)) if length == "oracle" else None)
            rows.append(evaluate_summary(d.sentences, d.labels, mask))
        return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
