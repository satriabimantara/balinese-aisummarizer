import json

import numpy as np
import pytest

from satuasummarization.extractive import (FEATURE_NAMES, Document, MetaheuristicSummarizer, SentenceFeatureExtractor,
                                           available_models, evaluate_summary, rouge_l, rouge_n, split_sentences,
                                           train_test_split_docs)
from satuasummarization.extractive.metrics import FastRouge

TEXT = ("I Kancil ngalih amah-amahan di alase. Ia nepukin I Kakua di sisin tukade. "
        "\"Kakua, cai lakar kija?\" keto I Kancil nakonang. I Kakua masaut ia lakar ngalih woh-wohan. "
        "Makadadua lantas majalan bareng ka tengah alase. Di tengah alase ada punyan poh liu pesan buahne.")
SENTS = split_sentences(TEXT)


@pytest.fixture(scope="module")
def lexical():
    return SentenceFeatureExtractor(linguistic=False)


def doc(fx, labels=(1, 1, 0, 0, 1, 0)):
    return Document("I Kancil", SENTS, np.array(labels), "test", "unit", fx.transform(SENTS, "I Kancil"))


def test_split_and_features_shape(lexical):
    assert len(SENTS) == 6
    F = lexical.transform(SENTS, title="I Kancil")
    assert F.shape == (6, len(FEATURE_NAMES)) and np.all((F >= 0) & (F <= 1))
    assert F[0, FEATURE_NAMES.index("sentence_position")] == 1
    assert F[2, FEATURE_NAMES.index("dialogue_ratio")] > 0       # direct speech sentence
    with pytest.raises(ValueError):
        lexical.transform([])


def test_rouge_known_values():
    assert rouge_n("a b c", "a b c")[2] == 1.0
    p, r, f = rouge_n("a b", "a b c d", 1)
    assert (p, r) == (1.0, 0.5) and abs(f - 2 / 3) < 1e-12
    assert rouge_n("x y", "a b")[2] == 0.0
    assert rouge_l("a x b c", "a b c")[1] == 1.0
    fr = FastRouge(["a b c", "d e", "f"], [1, 0, 0])
    r1, r2 = fr.f1(np.array([[1, 0, 0], [0, 1, 0]], bool))
    assert r1.tolist() == [1.0, 0.0] and r2[0] == 1.0


def test_fastrouge_matches_reference_implementation():
    sents = ["satu dua tiga", "dua tiga empat lima", "enam tujuh", "satu enam"]
    labels = [1, 0, 1, 0]
    mask = np.array([1, 1, 0, 0], bool)
    r1, r2 = FastRouge(sents, labels).f1(mask[None])
    ev = evaluate_summary(sents, labels, mask)
    assert abs(r1[0] - ev["rouge1_f1"]) < 1e-12 and abs(r2[0] - ev["rouge2_f1"]) < 1e-12


def test_fit_select_save_load(tmp_path, lexical):
    d = doc(lexical)
    m = MetaheuristicSummarizer(ratio=0.5).fit([d], algorithm="GWO", n_agents=8, max_iter=10, seed=0)
    assert m.weights.shape == (len(FEATURE_NAMES),) and m.meta["train_fitness"] > 0
    assert m.select(d.features, k=3).sum() == 3
    m2 =MetaheuristicSummarizer.load(m.save(tmp_path / "w.json"), extractor=lexical)
    assert np.allclose(m.weights, m2.weights)
    res = m2.summarize(TEXT, title="I Kancil", ratio=0.5)
    assert sum(s["selected"] for s in res["sentences"]) == 3
    assert res["summary"].startswith(next(s["text"] for s in res["sentences"] if s["selected"]))
    assert m2.evaluate([d])["rouge1_f1"] > 0


def test_invalid_inputs(lexical):
    with pytest.raises(ValueError):
        MetaheuristicSummarizer(weights=[1, 2])
    with pytest.raises(ValueError):
        MetaheuristicSummarizer(ratio=0)
    with pytest.raises(RuntimeError):
        MetaheuristicSummarizer().select(np.zeros((3, len(FEATURE_NAMES))))
    m = MetaheuristicSummarizer(np.ones(len(FEATURE_NAMES)), extractor=lexical)
    with pytest.raises(ValueError):
        m.summarize("   ")
    with pytest.raises(ValueError):
        m.summarize(TEXT, ratio=1.5)


def test_split_documents_by_source():
    docs = [Document(str(i), ["a b"], np.array([1]), "s1" if i < 10 else "s2") for i in range(20)]
    tr, te = train_test_split_docs(docs, 0.2, seed=1)
    assert len(te) == 4 and not {d.title for d in tr} & {d.title for d in te}


def test_pretrained_bundle_integrity(lexical):
    index = available_models()
    assert index["best"] in index["models"]
    for name in index["models"]:
        m = MetaheuristicSummarizer.from_pretrained(name, extractor=lexical)
        assert m.weights.shape == (len(FEATURE_NAMES),) and np.all((m.weights >= 0) & (m.weights <= 1))
    best = MetaheuristicSummarizer.from_pretrained("best", extractor=lexical)
    assert best.summarize(TEXT, title="I Kancil")["summary"]
    with pytest.raises(ValueError):
        MetaheuristicSummarizer.from_pretrained("NOPE")


def test_tampered_file_is_rejected(tmp_path, monkeypatch, lexical):
    from satuasummarization.extractive import summarizer as sm
    index = available_models()
    name = index["best"]
    (tmp_path / "index.json").write_text(json.dumps(index))
    raw = json.loads(sm._pretrained_root().joinpath(index["models"][name]["file"]).read_text())
    raw["weights"][0] = 0.123
    (tmp_path / index["models"][name]["file"]).write_text(json.dumps(raw))
    monkeypatch.setattr(sm, "_pretrained_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        MetaheuristicSummarizer.from_pretrained(name, extractor=lexical)


def test_linguistic_features_when_available():
    pytest.importorskip("balinese_nlp")
    F = SentenceFeatureExtractor().transform(SENTS, "I Kancil")
    assert F[:, FEATURE_NAMES.index("noun_density")].max() == 1


def test_double_spaces_do_not_crash_the_linguistic_features():
    pytest.importorskip("balinese_nlp")
    fx = SentenceFeatureExtractor()
    sents = ["Satios ring pasisi, genah sane becik taler karauhin  wantah ring gunung.", "I Kancil  ngalih  amah-amahan."]
    F = fx.transform(sents, "Segara Gunung")
    assert F.shape == (2, len(FEATURE_NAMES)) and fx.linguistic_failures_ == 0
