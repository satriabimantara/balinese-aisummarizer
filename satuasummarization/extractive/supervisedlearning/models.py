"""Sentence classifiers for extractive summarisation (label 1 = the sentence belongs to the summary).

Every model follows the same small contract::

    model.fit(X, y, groups)            # groups[i] = document of sentence i
    model.predict_proba(X, groups)     # (n, 2) array, column 1 = P(selected)

scikit-learn / XGBoost models ignore ``groups``. The two PyTorch models use it: both hold out 10% of the
training documents for early stopping, and ``AttentionNet`` reads every document as one sequence so a sentence
is classified in the context of the other sentences. Hyper-parameters are library defaults unless listed in
``MODEL_INFO``.
"""
import copy
import math

import numpy as np

# name -> (family, short description, non-default hyper-parameters)
MODEL_INFO = {
    "KNN": ("nearest neighbor", "k-nearest neighbours, k = 5", {}),
    "DecisionTree": ("decision tree", "CART, Gini impurity, unlimited depth", {}),
    "GaussianNB": ("naive bayes", "Gaussian naive Bayes", {}),
    "LogisticRegression": ("linear", "L2 logistic regression, C = 1", {"max_iter": 1000}),
    "SVM": ("kernel", "RBF support vector machine, C = 1, gamma = scale", {"probability": True}),
    "RandomForest": ("ensemble", "bagging of 100 trees", {}),
    "ExtraTrees": ("ensemble", "100 extremely randomised trees", {}),
    "AdaBoost": ("ensemble", "50 boosted decision stumps", {}),
    "GradientBoosting": ("ensemble", "100 boosted depth-3 trees, learning rate 0.1", {}),
    "HistGradientBoosting": ("ensemble", "histogram gradient boosting, 100 iterations", {}),
    "XGBoost": ("ensemble", "XGBoost, 100 trees, depth 6, eta 0.3", {}),
    "MLP": ("neural network", "scikit-learn MLP, one hidden layer of 100 ReLU units, Adam", {}),
    "DeepMLP": ("deep learning", "feed-forward network 128-64-32, ReLU, dropout 0.2",
                {"hidden": [128, 64, 32], "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 100,
                 "patience": 10}),
    "AttentionNet": ("deep learning", "2-layer Transformer encoder over the sentences of a document "
                     "(d_model 64, 4 heads, feed-forward 128, dropout 0.1)",
                     {"d_model": 64, "heads": 4, "layers": 2, "ff": 128, "dropout": 0.1, "lr": 1e-3,
                      "batch_size": 16, "epochs": 100, "patience": 10}),
}
MODEL_NAMES = list(MODEL_INFO)
TORCH_MODELS = ("DeepMLP", "AttentionNet")


def build_model(name, seed=42):
    """A new, unfitted model. scikit-learn estimators sit behind a StandardScaler."""
    if name not in MODEL_INFO:
        raise ValueError(f"unknown model {name!r}; available: {MODEL_NAMES}")
    params = MODEL_INFO[name][2]
    if name in TORCH_MODELS:
        return TorchSentenceClassifier(arch=name, seed=seed, **params)
    from sklearn import ensemble, linear_model, naive_bayes, neighbors, neural_network, svm, tree
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "XGBoost":
        from xgboost import XGBClassifier
        est = XGBClassifier(random_state=seed, n_jobs=1)
    else:
        cls = {"KNN": neighbors.KNeighborsClassifier, "DecisionTree": tree.DecisionTreeClassifier,
               "GaussianNB": naive_bayes.GaussianNB, "LogisticRegression": linear_model.LogisticRegression,
               "SVM": svm.SVC, "RandomForest": ensemble.RandomForestClassifier,
               "ExtraTrees": ensemble.ExtraTreesClassifier, "AdaBoost": ensemble.AdaBoostClassifier,
               "GradientBoosting": ensemble.GradientBoostingClassifier,
               "HistGradientBoosting": ensemble.HistGradientBoostingClassifier,
               "MLP": neural_network.MLPClassifier}[name]
        est = cls(**params)
        if "random_state" in est.get_params():
            est.set_params(random_state=seed)
    return _GroupBlind(make_pipeline(StandardScaler(), est))


class _GroupBlind:
    """Gives a scikit-learn estimator the (X, y, groups) contract."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, groups=None):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X, groups=None):
        return self.estimator.predict_proba(X)


def _document_slices(groups, n):
    """Row indices of every document, in order of first appearance."""
    if groups is None:
        return [np.arange(n)]
    groups = np.asarray(groups)
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    order = np.argsort(first)
    return [np.flatnonzero(inverse == g) for g in order]


def _network(arch, n_features, p):
    from torch import nn

    if arch == "DeepMLP":
        layers, width = [], n_features
        for h in p["hidden"]:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p["dropout"])]
            width = h
        body = nn.Sequential(*layers, nn.Linear(width, 1))

        class DeepMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = body

            def forward(self, x, mask=None):
                return self.body(x).squeeze(-1)

        return DeepMLP()

    import torch

    class AttentionNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = p["d_model"]
            self.embed = nn.Linear(n_features, d)
            layer = nn.TransformerEncoderLayer(d, p["heads"], p["ff"], p["dropout"], batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, p["layers"], enable_nested_tensor=False)
            self.head = nn.Linear(d, 1)

        def forward(self, x, mask):                       # x (docs, sentences, features); mask True = sentence
            d, length = self.embed.out_features, x.shape[1]
            pos = torch.arange(length, device=x.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2, device=x.device) * (-math.log(10000.0) / d))
            pe = torch.zeros(length, d, device=x.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            h = self.encoder(self.embed(x) + pe, src_key_padding_mask=~mask)
            return self.head(h).squeeze(-1)

    return AttentionNet()


class TorchSentenceClassifier:
    """DeepMLP (sentences classified independently) or AttentionNet (document = sequence of sentences).

    Adam + binary cross-entropy; 10% of the training documents are held out and training stops when their loss
    has not improved for ``patience`` epochs (the best epoch is restored). ``history_`` keeps both loss curves."""

    def __init__(self, arch="DeepMLP", lr=1e-3, batch_size=64, epochs=100, patience=10, seed=42, **arch_params):
        if arch not in TORCH_MODELS:
            raise ValueError(f"arch must be one of {TORCH_MODELS}")
        self.arch, self.lr, self.batch_size, self.epochs, self.patience, self.seed = arch, lr, batch_size, epochs, patience, seed
        self.arch_params = arch_params
        self.net_ = None

    # ------------------------------------------------------------------ batching
    def _batches(self, X, y, slices, shuffle, rng):
        """Lists of (x, y, mask) tensors: rows for DeepMLP, padded documents for AttentionNet."""
        import torch

        if self.arch == "DeepMLP":
            idx = np.concatenate(slices)
            if shuffle:
                idx = rng.permutation(idx)
            for i in range(0, len(idx), self.batch_size):
                b = idx[i:i + self.batch_size]
                yield (torch.as_tensor(X[b], dtype=torch.float32), torch.as_tensor(y[b], dtype=torch.float32),
                       torch.ones(len(b), dtype=torch.bool))
            return
        order = rng.permutation(len(slices)) if shuffle else np.arange(len(slices))
        for i in range(0, len(order), self.batch_size):
            docs = [slices[j] for j in order[i:i + self.batch_size]]
            length = max(len(d) for d in docs)
            xb = np.zeros((len(docs), length, X.shape[1]), dtype=np.float32)
            yb = np.zeros((len(docs), length), dtype=np.float32)
            mb = np.zeros((len(docs), length), dtype=bool)
            for r, d in enumerate(docs):
                xb[r, :len(d)], yb[r, :len(d)], mb[r, :len(d)] = X[d], y[d], True
            yield torch.as_tensor(xb), torch.as_tensor(yb), torch.as_tensor(mb)

    def _loss(self, slices, X, y, train, rng, opt=None):
        import torch

        self.net_.train(train)
        total, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, yb, mb in self._batches(X, y, slices, train, rng):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net_(xb, mb)[mb], yb[mb])
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total, count = total + loss.item() * int(mb.sum()), count + int(mb.sum())
        return total / count

    # ------------------------------------------------------------------ API
    def fit(self, X, y, groups=None):
        import torch

        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if groups is None:
            raise ValueError("groups (document id of every sentence) is required to train")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        slices = _document_slices(groups, len(X))
        if len(slices) < 2:
            raise ValueError("need at least 2 documents (one is held out for early stopping)")
        perm = rng.permutation(len(slices))
        n_val = max(1, round(0.1 * len(slices)))
        val, train = [slices[i] for i in perm[:n_val]], [slices[i] for i in perm[n_val:]]
        self.n_features_ = X.shape[1]
        self.net_ = _network(self.arch, self.n_features_, self.arch_params)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.history_ = {"train_loss": [], "val_loss": []}
        best, best_state, wait = math.inf, None, 0
        for _ in range(self.epochs):
            self.history_["train_loss"].append(self._loss(train, X, y, True, rng, opt))
            v = self._loss(val, X, y, False, rng)
            self.history_["val_loss"].append(v)
            if v < best - 1e-5:
                best, best_state, wait = v, copy.deepcopy(self.net_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        return self

    def predict_proba(self, X, groups=None):
        import torch

        if self.net_ is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        slices = _document_slices(groups, len(X))
        p = np.zeros(len(X))
        y = np.zeros(len(X), dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            if self.arch == "DeepMLP":
                p[:] = torch.sigmoid(self.net_(torch.as_tensor(X))).numpy()
            else:
                for d in slices:                      # one document per forward pass: no padding at inference
                    xb, _, mb = next(self._batches(X, y, [d], False, None))
                    p[d] = torch.sigmoid(self.net_(xb, mb))[0].numpy()
        return np.column_stack([1 - p, p])

    # ------------------------------------------------------------------ persistence (arrays only, no pickle)
    def state(self):
        """(config dict, {parameter name: numpy array})."""
        config = {"arch": self.arch, "n_features": self.n_features_, "lr": self.lr, "batch_size": self.batch_size,
                  "epochs": self.epochs, "patience": self.patience, "seed": self.seed, "arch_params": self.arch_params}
        return config, {k: v.numpy() for k, v in self.net_.state_dict().items()}

    @classmethod
    def from_state(cls, config, arrays):
        import torch

        m = cls(config["arch"], config["lr"], config["batch_size"], config["epochs"], config["patience"],
                config["seed"], **config["arch_params"])
        m.n_features_ = config["n_features"]
        m.net_ = _network(m.arch, m.n_features_, m.arch_params)
        m.net_.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in arrays.items()})
        return m
