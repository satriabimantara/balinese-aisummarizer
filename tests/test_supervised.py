import json

import numpy as np
import pytest

from satuasummarization.extractive import (FEATURE_NAMES, Document, SentenceFeatureExtractor, SupervisedSummarizer,
                                           available_supervised_models, bleu, embedding_similarity, evaluate_summary,
                                           split_sentences)
from satuasummarization.extractive.supervisedlearning import MODEL_NAMES, build_model
from satuasummarization.extractive.supervisedlearning.models import _document_slices

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

TEXT = ("I Kancil ngalih amah-amahan di alase. Ia nepukin I Kakua di sisin tukade. "
        "\"Kakua, cai lakar kija?\" keto I Kancil nakonang. I Kakua masaut ia lakar ngalih woh-wohan. "
        "Makadadua lantas majalan bareng ka tengah alase. Di tengah alase ada punyan poh liu pesan buahne.")
SENTS = split_sentences(TEXT)


@pytest.fixture(scope="module")
def lexical():
    return SentenceFeatureExtractor(linguistic=False)


@pytest.fixture(scope="module")
def docs(lexical):
    F = lexical.transform(SENTS, "I Kancil")
    rng = np.random.default_rng(0)
    out = []
    for i in range(12):                       # label = "early sentence", learnable from sentence_position
        order = rng.permutation(len(SENTS))
        out.append(Document(f"d{i}", [SENTS[j] for j in order], (F[order, 1] > 0.5).astype(int), "test", "unit", F[order]))
    return out


def test_bleu_and_embedding_similarity():
    assert bleu("a b c d e", "a b c d e") == pytest.approx(1.0)
    assert bleu("x y z", "a b c") == 0.0 and bleu("", "a b") == 0.0
    assert bleu("a b c", "a b c d e f") < bleu("a b c d e", "a b c d e f")          # brevity penalty
    nltk = pytest.importorskip("nltk.translate.bleu_score")
    a, b = "i kancil ngalih amah-amahan di alase ia nepukin i kakua", "i kancil ngalih woh-wohan di alase tur nepukin i kakua"
    ref = nltk.sentence_bleu([b.split()], a.split(), smoothing_function=nltk.SmoothingFunction().method2)
    assert bleu(a, b) == pytest.approx(ref)
    E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert embedding_similarity(E[[0]], E[[0]]) == pytest.approx(1.0)
    assert embedding_similarity(E[[0]], E[[1]]) == 0.0
    ev = evaluate_summary(["a b", "c d", "e f"], [1, 0, 1], [1, 1, 0], embeddings=E)
    assert 0 < ev["bleu"] < 1 and 0 < ev["embedding_similarity"] < 1 and ev["rouge1_recall"] == 0.5


def test_document_slices_keep_document_order():
    slices = _document_slices(np.array([7, 7, 2, 2, 2, 9]), 6)
    assert [s.tolist() for s in slices] == [[0, 1], [2, 3, 4], [5]]
    assert _document_slices(None, 3)[0].tolist() == [0, 1, 2]


@pytest.mark.parametrize("name", ["GaussianNB", "DecisionTree", "DeepMLP", "AttentionNet"])
def test_fit_select_save_load(tmp_path, lexical, docs, name):
    m = SupervisedSummarizer().fit(docs, model=name, seed=0)
    d = docs[0]
    p = m.predict_proba(d.features)
    assert p.shape == (len(SENTS),) and np.all((p >= 0) & (p <= 1))
    assert m.select(d.features, k=2).sum() == 2 and m.select(d.features, ratio=0.5).sum() == 3
    assert m.select(d.features, threshold=0.999999).sum() >= 1                    # never an empty summary
    path = m.save(tmp_path / "model")
    assert path.suffix == (".npz" if name in ("DeepMLP", "AttentionNet") else ".joblib")
    m2 = SupervisedSummarizer.load(path, extractor=lexical)
    assert m2.name == name and np.allclose(p, m2.predict_proba(d.features), atol=1e-6)
    res = m2.summarize(TEXT, title="I Kancil", ratio=0.5)
    assert sum(s["selected"] for s in res["sentences"]) == 3 and res["algorithm"] == name and res["threshold"] is None
    res = m2.summarize(TEXT, title="I Kancil")
    assert res["ratio"] is None and res["threshold"] == 0.5 and res["summary"]
    assert m2.evaluate(docs, "oracle")["rouge1_f1"] > 0 and "bleu" in m2.evaluate(docs)


def test_classifier_learns_the_rule(docs):
    m = SupervisedSummarizer().fit(docs, model="DecisionTree")
    assert m.evaluate(docs)["f1_macro"] == 1.0


def test_invalid_inputs(lexical, docs):
    with pytest.raises(ValueError):
        SupervisedSummarizer(ratio=0)
    with pytest.raises(ValueError):
        SupervisedSummarizer(threshold=1)
    with pytest.raises(RuntimeError):
        SupervisedSummarizer().predict_proba(np.zeros((3, len(FEATURE_NAMES))))
    with pytest.raises(ValueError):
        build_model("NOPE")
    with pytest.raises(ValueError):
        SupervisedSummarizer().fit([])
    with pytest.raises(ValueError):
        build_model("DeepMLP").fit(np.zeros((4, 13)), np.array([0, 1, 0, 1]))     # groups are required
    m = SupervisedSummarizer(extractor=lexical).fit(docs, model="GaussianNB")
    with pytest.raises(ValueError):
        m.summarize("   ")
    with pytest.raises(ValueError):
        m.summarize(TEXT, ratio=1.5)
    with pytest.raises(ValueError):
        m.evaluate(docs, length="ratio")


def test_file_for_other_features_is_rejected(tmp_path, docs, monkeypatch):
    from satuasummarization.extractive.supervisedlearning import summarizer as sm

    path = SupervisedSummarizer().fit(docs, model="DeepMLP").save(tmp_path / "m")
    monkeypatch.setattr(sm, "FEATURE_NAMES", FEATURE_NAMES[:-1])
    with pytest.raises(ValueError, match="different feature set"):
        SupervisedSummarizer.load(path)


def test_pretrained_bundle_integrity(lexical):
    index = available_supervised_models()
    assert index["best"] in index["models"] and set(index["models"]) == set(MODEL_NAMES)
    for name in index["models"]:
        m = SupervisedSummarizer.from_pretrained(name, extractor=lexical)
        out = m.summarize(TEXT, title="I Kancil")
        assert out["summary"] and out["algorithm"] == name
    with pytest.raises(ValueError):
        SupervisedSummarizer.from_pretrained("NOPE")


def test_tampered_file_is_rejected(tmp_path, monkeypatch, lexical):
    from satuasummarization.extractive.supervisedlearning import summarizer as sm

    index = available_supervised_models()
    entry = index["models"][index["best"]]
    (tmp_path / "index.json").write_text(json.dumps(index))
    raw = bytearray(sm._pretrained_root().joinpath(entry["file"]).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (tmp_path / entry["file"]).write_bytes(bytes(raw))
    monkeypatch.setattr(sm, "_pretrained_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        SupervisedSummarizer.from_pretrained("best", extractor=lexical)
