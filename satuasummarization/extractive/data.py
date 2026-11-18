"""Loader for the BaliSummarization extractive corpus (one .xlsx per document with the columns
``sentences`` and ``extractive_summary``; the file name is the title)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import tokenize


@dataclass
class Document:
    title: str
    sentences: list
    labels: np.ndarray
    source: str = ""
    subset: str = ""
    features: np.ndarray = field(default=None, repr=False)


def load_corpus(root, subsets=("complete", "independen", "pilot", "pre")):
    """Read every document under ``root/<source>/<subset>/*.xlsx``. Rows without any word token
    (stray quote marks) are dropped together with their label."""
    import pandas as pd

    docs = []
    for path in sorted(Path(root).glob("*/*/*.xlsx")):
        source, subset = path.parent.parent.name, path.parent.name
        if subset not in subsets:
            continue
        df = pd.read_excel(path)
        if not {"sentences", "extractive_summary"} <= set(df.columns):
            raise ValueError(f"{path}: expected columns 'sentences' and 'extractive_summary'")
        df = df[df["sentences"].astype(str).map(lambda s: bool(tokenize(s)))]
        labels = df["extractive_summary"].fillna(0).astype(int).clip(0, 1).to_numpy()
        if len(df) < 2 or labels.sum() == 0:
            continue
        docs.append(Document(path.stem, df["sentences"].astype(str).str.strip().tolist(), labels, source, subset))
    return docs


def train_test_split_docs(docs, test_size=0.2, seed=42):
    """Split documents (not sentences) with the same proportion inside every source."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for source in sorted({d.source for d in docs}):
        group = [d for d in docs if d.source == source]
        idx = rng.permutation(len(group))
        k = max(1, int(round(test_size * len(group))))
        test += [group[i] for i in idx[:k]]
        train += [group[i] for i in idx[k:]]
    return train, test
