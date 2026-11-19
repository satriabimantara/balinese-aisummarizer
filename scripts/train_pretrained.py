"""Train the pretrained optimal weights of every metaheuristic and write
satuasummarization/extractive/metaheuristics/pretrained_optimal_models/{<ALG>.json, index.json} + results/*.csv.

python scripts/train_pretrained.py [--agents 30 --iters 100 --seeds 3 --jobs 12] [ALG ...]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from satuasummarization.extractive import FEATURE_NAMES, MetaheuristicSummarizer, train_test_split_docs  # noqa: E402
from satuasummarization.extractive.metrics import evaluate_summary  # noqa: E402
from satuasummarization.extractive.summarizer import content_sha256  # noqa: E402

OUT = ROOT / "satuasummarization/extractive/metaheuristics/pretrained_optimal_models"
RES = ROOT / "results"


def run(alg, train, seed, agents, iters):
    m = MetaheuristicSummarizer(ratio=0.4).fit(train, algorithm=alg, n_agents=agents, max_iter=iters, seed=seed)
    return alg, seed, m


def cv_fold(alg, fit_docs, val_docs, fold, agents, iters):
    """Model selection: fit on 4 folds of the training split, score on the held-out fold (never on test)."""
    m = MetaheuristicSummarizer(ratio=0.4).fit(fit_docs, algorithm=alg, n_agents=agents, max_iter=iters, seed=fold)
    return alg, fold, m.evaluate(val_docs, length="oracle")


def _score(d, prefix=""):
    return (d[f"{prefix}rouge1_f1"] + d[f"{prefix}rouge2_f1"] + d[f"{prefix}rougeL_f1"]) / 3


def baseline_rows(test):
    rows = {}
    n_feat = len(FEATURE_NAMES)
    variants = {"Lead-k": None, "EqualWeights": np.ones(n_feat), "RandomWeights": np.random.default_rng(0).random(n_feat)}
    for name, w in variants.items():
        res = []
        for d in test:
            k = int(d.labels.sum())
            if w is None:
                mask = np.zeros(len(d.sentences), bool)
                mask[:k] = True
            else:
                mask = MetaheuristicSummarizer(w).select(d.features, k=k)
            res.append(evaluate_summary(d.sentences, d.labels, mask))
        rows[name] = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algorithms", nargs="*")
    ap.add_argument("--agents", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    from metaheuristicpy.core import ALGORITHMS

    algs = a.algorithms or list(ALGORITHMS)
    docs = pickle.load(open(RES / "corpus_features.pkl", "rb"))
    train, test = train_test_split_docs(docs, test_size=0.2, seed=42)
    print(f"{len(train)} train docs, {len(test)} test docs; {len(algs)} algorithms x {a.seeds} seeds", flush=True)

    from sklearn.model_selection import KFold

    folds = list(KFold(a.folds, shuffle=True, random_state=42).split(np.arange(len(train))))
    t0 = time.time()
    jobs = [delayed(cv_fold)(alg, [train[i] for i in fi], [train[i] for i in vi], k, a.agents, a.iters)
            for alg in algs for k, (fi, vi) in enumerate(folds)]
    jobs += [delayed(run)(alg, train, s, a.agents, a.iters) for alg in algs for s in range(a.seeds)]
    out = Parallel(n_jobs=a.jobs, verbose=5)(jobs)
    cv_runs, runs = out[: len(algs) * a.folds], out[len(algs) * a.folds:]
    print(f"cross-validation + training done in {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"models": {}}
    rows, curves = [], {}
    for alg in algs:
        cv = [r for g, _, r in cv_runs if g == alg]
        cv_mean = {k: float(np.mean([r[k] for r in cv])) for k in cv[0]}
        cv_std = {k: float(np.std([r[k] for r in cv])) for k in cv[0]}
        mine = [m for g, _, m in runs if g == alg]
        fits = [m.meta["train_fitness"] for m in mine]
        best = mine[int(np.argmax(fits))]
        oracle = best.evaluate(test, length="oracle")
        ratio = best.evaluate(test, length="ratio")
        best.meta.update(cv_mean=cv_mean, cv_std=cv_std, test_oracle_length=oracle, test_ratio_0_4=ratio,
                         train_fitness_seeds=fits,
                         split={"test_size": 0.2, "seed": 42, "n_train": len(train), "n_test": len(test),
                                "cv_folds": a.folds, "cv_seed": 42})
        path = OUT / f"{alg}.json"
        best.save(path)
        index["models"][alg] = {"file": path.name, "sha256": content_sha256(best.to_dict()),
                                "test_rouge1_f1": oracle["rouge1_f1"], "test_rouge2_f1": oracle["rouge2_f1"],
                                "test_rougeL_f1": oracle["rougeL_f1"], "test_f1_macro": oracle["f1_macro"],
                                "cv_rouge1_f1": cv_mean["rouge1_f1"], "cv_rouge2_f1": cv_mean["rouge2_f1"],
                                "cv_rougeL_f1": cv_mean["rougeL_f1"], "cv_score": _score(cv_mean),
                                "cv_score_std": float(np.std([_score(r) for r in cv])),
                                "train_fitness": max(fits)}
        rows.append({"algorithm": alg, "train_fitness_mean": np.mean(fits), "train_fitness_std": np.std(fits),
                     "train_time_s": np.mean([m.meta["train_time_s"] for m in mine]),
                     "cv_score": _score(cv_mean), **{f"cv_{k}": v for k, v in cv_mean.items()},
                     **{f"test_{k}": v for k, v in oracle.items()}})
        curves[alg] = [m.meta["history"] for m in mine]
    index["best"] = max(index["models"], key=lambda k: index["models"][k]["cv_score"])
    index["selection_rule"] = (f"highest mean ROUGE-1/2/L F1 over {a.folds}-fold cross-validation inside the training "
                               "split (80%); the test split (20%) is only used for reporting")
    index["feature_names"] = FEATURE_NAMES
    index_path.write_text(json.dumps(index, indent=2), newline="\n")

    for name, r in baseline_rows(test).items():
        rows.append({"algorithm": name, **{f"test_{k}": v for k, v in r.items()}})
    df = pd.DataFrame(rows)
    old = RES / "pretrained_results.csv"
    if a.algorithms and old.exists():
        df = pd.concat([pd.read_csv(old).query("algorithm not in @df.algorithm"), df])
    df.sort_values("cv_score", ascending=False, na_position="last").to_csv(old, index=False)
    conv_path = RES / "pretrained_convergence.json"
    if a.algorithms and conv_path.exists():
        curves = {**json.load(open(conv_path)), **curves}
    json.dump(curves, open(conv_path, "w"))
    print(df.sort_values("cv_score", ascending=False)[["algorithm", "cv_score", "test_rouge1_f1"]].head(15).to_string(index=False))
    print("best:", index["best"])


if __name__ == "__main__":
    main()
