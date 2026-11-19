"""Extractive Balinese summarisation.

Two paradigms share the same 13 sentence features (``SentenceFeatureExtractor``):

* ``metaheuristics``     - score(s) = w . x(s), weights found by a metaheuristic (ROUGE fitness)
* ``supervisedlearning`` - binary sentence classification (1 = sentence belongs to the summary)

>>> from satuasummarization.extractive import MetaheuristicSummarizer, SupervisedSummarizer
>>> MetaheuristicSummarizer.from_pretrained("best").summarize(text, title="I Kancil", ratio=0.3)["summary"]
>>> SupervisedSummarizer.from_pretrained("best").summarize(text, title="I Kancil")["summary"]
"""
from .data import Document, load_corpus, train_test_split_docs
from .features import FEATURE_NAMES, SentenceFeatureExtractor, split_sentences, tokenize
from .metaheuristics import MetaheuristicSummarizer, available_models
from .metrics import bleu, embedding_similarity, evaluate_summary, rouge_l, rouge_n
from .supervisedlearning import SupervisedSummarizer, available_supervised_models

__all__ = ["Document", "FEATURE_NAMES", "MetaheuristicSummarizer", "SentenceFeatureExtractor", "SupervisedSummarizer",
           "available_models", "available_supervised_models", "bleu", "embedding_similarity", "evaluate_summary",
           "load_corpus", "rouge_l", "rouge_n", "split_sentences", "tokenize", "train_test_split_docs"]
