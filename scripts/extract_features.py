"""Extract the 13 sentence features of every corpus document once and cache them (features are deterministic)."""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from satuasummarization.extractive import SentenceFeatureExtractor, load_corpus  # noqa: E402

docs = load_corpus(ROOT / "datasets/BalineseSummarizationDatasets/extractive")
fx = SentenceFeatureExtractor()
t = time.time()
for i, d in enumerate(docs):
    d.features = fx.transform(d.sentences, d.title)
    if i % 25 == 0:
        print(f"{i}/{len(docs)} {time.time() - t:.0f}s", flush=True)
out = ROOT / "results/corpus_features.pkl"
pickle.dump(docs, open(out, "wb"))
print("saved", out, len(docs), "docs", sum(len(d.sentences) for d in docs), "sentences")
