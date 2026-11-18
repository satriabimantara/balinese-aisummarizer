"""Sentence feature extraction for Balinese extractive summarisation.

Each sentence of a document is described by 13 features, min-max normalised inside the document so that
weights learned on one corpus transfer to documents of any length:

==  =====================  =====================================================================
#   name                   definition
==  =====================  =====================================================================
1   sentence_length        tokens in the sentence / tokens in the longest sentence
2   sentence_position      1 - index / n  (earlier sentences score higher)
3   numerical_ratio        share of numeric tokens
4   title_jaccard          Jaccard overlap between sentence and title vocabularies
5   tfisf_mean             mean TF-ISF weight of the sentence tokens (ISF computed in the document)
6   centroid_cosine        cosine(TF-ISF sentence vector, TF-ISF document centroid)
7   title_cosine           cosine(TF-ISF sentence vector, TF-ISF title vector)
8   intra_similarity       mean cosine to the other sentences of the document
9   dialogue_ratio         share of tokens inside quotation marks (direct speech)
10  noun_density           NN/NND tags / tokens  (balinese_nlp HMM POS tagger)
11  verb_density           VB/MD tags / tokens
12  propernoun_density     NNP tags / tokens
13  character_density      characters found by SatuaNER / tokens
==  =====================  =====================================================================

Features 10-13 need ``balinese-nlp`` (``pip install satuasummarization[nlp]``); with
``linguistic=False`` they are returned as zeros (the pretrained weights expect all 13).
"""
import re

import numpy as np

FEATURE_NAMES = [
    "sentence_length", "sentence_position", "numerical_ratio", "title_jaccard", "tfisf_mean",
    "centroid_cosine", "title_cosine", "intra_similarity", "dialogue_ratio",
    "noun_density", "verb_density", "propernoun_density", "character_density",
]
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_QUOTE = re.compile(r"(``|''|\"|“|”)")


def tokenize(text):
    """Lower-cased word tokens (letters/digits, hyphenated words kept together)."""
    return _TOKEN.findall(str(text).lower())


def split_sentences(text):
    """Rule-based sentence splitter for raw Balinese text (., !, ? followed by space and a capital,
    quote or digit; newlines always split)."""
    parts = []
    for block in re.split(r"\n\s*\n|\r?\n", str(text)):
        block = block.strip()
        if block:
            parts.extend(s.strip() for s in re.split(r"(?<=[.!?])[\"'”’]*\s+(?=[\"'“‘(`]*[A-Z0-9])", block) if s.strip())
    return [s for s in parts if tokenize(s)]


def _minmax(col):
    lo, hi = col.min(), col.max()
    return np.zeros_like(col) if hi - lo < 1e-12 else (col - lo) / (hi - lo)


class SentenceFeatureExtractor:
    """Extract the 13 summarisation features of every sentence of one document.

    >>> fx = SentenceFeatureExtractor()
    >>> X = fx.transform(sentences, title="I Kancil")   # shape (n_sentences, 13)
    """

    feature_names = FEATURE_NAMES

    def __init__(self, linguistic=True, character_ner=True):
        self.linguistic = linguistic
        self.character_ner = character_ner and linguistic
        self._pos = self._ner = None
        if linguistic:
            try:
                from balinese_nlp.postag.HMM import HiddenMarkovModelPOSTag
            except ImportError as exc:
                raise ImportError("linguistic features need balinese-nlp: pip install satuasummarization[nlp] "
                                  "or use SentenceFeatureExtractor(linguistic=False)") from exc
            self._pos = HiddenMarkovModelPOSTag()
            if self.character_ner:
                from balinese_nlp.narratives.utils import load_pretrained_character_ner_model
                self._ner = load_pretrained_character_ner_model(modelversion="satuaner")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tfisf(token_lists, title_tokens):
        vocab = sorted({t for toks in token_lists for t in toks} | set(title_tokens))
        index = {t: i for i, t in enumerate(vocab)}
        n = len(token_lists)
        tf = np.zeros((n, len(vocab)))
        for i, toks in enumerate(token_lists):
            for t in toks:
                tf[i, index[t]] += 1
        df = (tf > 0).sum(0)
        isf = np.log((n + 1) / (df + 1)) + 1
        S = tf * isf
        T = np.zeros(len(vocab))
        for t in title_tokens:
            T[index[t]] += 1
        return S, T * isf

    @staticmethod
    def _cos(A, b):
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(b)
        out = np.zeros(len(A))
        ok = (na > 0) & (nb > 0)
        out[ok] = A[ok] @ b / (na[ok] * nb)
        return out

    def _linguistic(self, sentences, n_tokens):
        out = np.zeros((len(sentences), 4))
        self.linguistic_failures_ = 0
        if not self.linguistic:
            return out
        for i, s in enumerate(sentences):
            s = " ".join(s.split())          # repeated spaces make empty tokens that crash the HMM tagger
            try:
                tagged = self._pos.predict(s).replace("\n", " ").split()
                tags = [tok.rsplit("/", 1)[-1] for tok in tagged if "/" in tok]
                m = max(len(tags), 1)
                out[i, 0] = sum(t in ("NN", "NND") for t in tags) / m
                out[i, 1] = sum(t in ("VB", "MD") for t in tags) / m
                out[i, 2] = sum(t == "NNP" for t in tags) / m
                if self._ner is not None:
                    y, _ = self._ner.predict_sentence(s)
                    chars = self._ner.extract_predicted_characters_from_sentence(s, y)["predicted_chars"]
                    if isinstance(chars, str) and chars:
                        out[i, 3] = len(chars.split("; ")) / max(n_tokens[i], 1)
            except (IndexError, KeyError, ValueError):
                # shortcut: a sentence balinese-nlp cannot tag keeps 0 linguistic features — revisit when the
                # tagger handles every token; counted in linguistic_failures_ so callers can see it
                out[i] = 0
                self.linguistic_failures_ += 1
        return out

    # ------------------------------------------------------------------ API
    def transform(self, sentences, title=""):
        """Return an (n_sentences, 13) array of document-normalised features."""
        sentences = [str(s) for s in sentences]
        if not sentences:
            raise ValueError("sentences must not be empty")
        toks = [tokenize(s) for s in sentences]
        n = len(sentences)
        n_tok = np.array([len(t) for t in toks], dtype=float)
        title_tokens = tokenize(title)
        title_set = set(title_tokens)
        S, T = self._tfisf(toks, title_tokens)

        F = np.zeros((n, len(FEATURE_NAMES)))
        F[:, 0] = n_tok / max(n_tok.max(), 1)
        F[:, 1] = 1 - np.arange(n) / n
        F[:, 2] = [sum(t.isdigit() for t in tk) / max(len(tk), 1) for tk in toks]
        F[:, 3] = [len(set(tk) & title_set) / len(set(tk) | title_set) if (tk or title_set) else 0 for tk in toks]
        F[:, 4] = S.sum(1) / np.maximum(n_tok, 1)
        F[:, 5] = self._cos(S, S.mean(0))
        F[:, 6] = self._cos(S, T)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        U = np.divide(S, norms, out=np.zeros_like(S), where=norms > 0)
        sim = U @ U.T
        F[:, 7] = (sim.sum(1) - np.diag(sim)) / max(n - 1, 1)
        F[:, 8] = [self._dialogue_ratio(s) for s in sentences]
        F[:, 9:13] = self._linguistic(sentences, n_tok)
        return np.column_stack([_minmax(F[:, j]) for j in range(F.shape[1])])

    @staticmethod
    def _dialogue_ratio(sentence):
        inside, count, total = False, 0, 0
        for piece in _QUOTE.split(sentence):
            if _QUOTE.fullmatch(piece or " "):
                inside = not inside
                continue
            k = len(tokenize(piece))
            total += k
            count += k if inside else 0
        return count / total if total else 0.0
