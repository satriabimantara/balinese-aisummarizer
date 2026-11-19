"""ROUGE-N / ROUGE-L and sentence-level metrics for extractive summaries (lower-cased tokens, no stemming)."""
from collections import Counter

import numpy as np

from .features import tokenize


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap, n_sys, n_ref):
    p = overlap / n_sys if n_sys else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rouge_n(system, reference, n=1):
    """(precision, recall, f1) of ROUGE-N between two texts."""
    s, r = _ngrams(tokenize(system), n), _ngrams(tokenize(reference), n)
    return _prf(sum((s & r).values()), sum(s.values()), sum(r.values()))


def rouge_l(system, reference):
    """(precision, recall, f1) of ROUGE-L (summary-level LCS on the concatenated texts)."""
    a, b = tokenize(system), tokenize(reference)
    if not a or not b:
        return 0.0, 0.0, 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return _prf(prev[-1], len(a), len(b))


def bleu(system, reference, max_n=4):
    """Sentence-level BLEU-4 with brevity penalty; n-gram precisions for n > 1 are add-one smoothed
    (Lin & Och, 2004), so a short summary without any 4-gram match does not score exactly 0."""
    s, r = tokenize(system), tokenize(reference)
    if not s or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        sg, rg = _ngrams(s, n), _ngrams(r, n)
        match, total = sum((sg & rg).values()), sum(sg.values())
        if n > 1:
            match, total = match + 1, total + 1
        if match == 0:
            return 0.0
        log_p += np.log(match / total) / max_n
    return float(min(1.0, np.exp(1 - len(r) / len(s))) * np.exp(log_p))


def embedding_similarity(system_vectors, reference_vectors):
    """Cosine similarity between the mean sentence embedding of the system and reference summaries."""
    a, b = np.mean(system_vectors, axis=0), np.mean(reference_vectors, axis=0)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0


def sentence_scores(y_true, y_pred):
    """Accuracy and macro precision/recall/F1 of the selected-sentence labels."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    out = {"accuracy": float((y_true == y_pred).mean())}
    ps, rs, fs = [], [], []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        p, r, f = _prf(tp, np.sum(y_pred == c), np.sum(y_true == c))
        ps.append(p), rs.append(r), fs.append(f)
    out.update(precision_macro=float(np.mean(ps)), recall_macro=float(np.mean(rs)), f1_macro=float(np.mean(fs)))
    return out


def _rouge_sentences(sys_sents, ref_sents, n):
    """ROUGE-N with n-grams counted inside sentences (no n-gram spans a sentence boundary)."""
    s = sum((_ngrams(tokenize(x), n) for x in sys_sents), Counter())
    r = sum((_ngrams(tokenize(x), n) for x in ref_sents), Counter())
    return _prf(sum((s & r).values()), sum(s.values()), sum(r.values()))


def evaluate_summary(sentences, y_true, y_pred, embeddings=None):
    """All metrics of one document: ROUGE-1/2/L (P/R/F1) and BLEU between selected and reference sentences,
    sentence-level accuracy/macro P/R/F1, and ``embedding_similarity`` when the (n_sentences, d) sentence
    ``embeddings`` are given."""
    sys_s = [s for s, k in zip(sentences, y_pred) if k]
    ref_s = [s for s, k in zip(sentences, y_true) if k]
    out = {"bleu": bleu(" ".join(sys_s), " ".join(ref_s))}
    if embeddings is not None and sys_s and ref_s:
        E = np.asarray(embeddings)
        out["embedding_similarity"] = embedding_similarity(E[np.asarray(y_pred, bool)], E[np.asarray(y_true, bool)])
    for key, (p, r, f) in {"rouge1": _rouge_sentences(sys_s, ref_s, 1), "rouge2": _rouge_sentences(sys_s, ref_s, 2),
                           "rougeL": rouge_l(" ".join(sys_s), " ".join(ref_s))}.items():
        out.update({f"{key}_precision": p, f"{key}_recall": r, f"{key}_f1": f})
    out.update(sentence_scores(y_true, y_pred))
    return out


class FastRouge:
    """Vectorised ROUGE-1/2 F1 of many candidate selections of one document (used as fitness).

    Precomputes the sentence x n-gram count matrices once; ``f1(masks)`` scores a (k, n_sentences)
    boolean matrix of selections against the reference labels in a few matrix operations."""

    def __init__(self, sentences, labels):
        toks = [tokenize(s) for s in sentences]
        self.mats, self.refs = [], []
        ref = np.asarray(labels, bool)
        for n in (1, 2):
            grams = [_ngrams(t, n) for t in toks]
            vocab = {g: i for i, g in enumerate({g for c in grams for g in c})}
            M = np.zeros((len(toks), max(len(vocab), 1)))
            for i, c in enumerate(grams):
                for g, k in c.items():
                    M[i, vocab[g]] = k
            self.mats.append(M)
            self.refs.append(M[ref].sum(0))

    def f1(self, masks):
        masks = np.atleast_2d(masks).astype(float)
        out = []
        for M, r in zip(self.mats, self.refs):
            sel = masks @ M
            overlap = np.minimum(sel, r).sum(1)
            n_sys, n_ref = sel.sum(1), r.sum()
            p = np.divide(overlap, n_sys, out=np.zeros_like(overlap), where=n_sys > 0)
            rc = overlap / n_ref if n_ref else np.zeros_like(overlap)
            out.append(np.divide(2 * p * rc, p + rc, out=np.zeros_like(p), where=(p + rc) > 0))
        return out  # [rouge1_f1, rouge2_f1], each of shape (k,)
