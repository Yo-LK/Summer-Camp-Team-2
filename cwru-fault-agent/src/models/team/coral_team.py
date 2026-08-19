"""
CORAL / TCA domain adaptation for CWRU cross-load fault diagnosis (role 4).

Runs the SAME tasks, splits, classifiers and seeds as baseline.py, so the only
difference between the two result sets is the adaptation step itself. That is the
whole point -- an improvement is only attributable to CORAL if nothing else moved.

Methods
-------
  none    baseline passthrough (sanity check: must reproduce baseline.py exactly)
  coral   CORrelation ALignment. Whiten source covariance, recolour with target's.
          Closed form, no training, no hyperparameters.
  tca     Transfer Component Analysis. Learn a kernel subspace where the MMD
          between source and target is minimised.

Usage
-----
  python coral.py --features features.csv --classes 10 --mode loo
  python coral.py --features features.csv --mode pairwise --methods none,coral
  python compare.py                      # merge with baseline and tabulate

Output
------
  results/adapt_raw_<mode>_<n>class.json
  results/adapt_summary_<mode>_<n>class.csv
"""
from __future__ import annotations

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

META_COLS = {
    "record_id", "window_number", "start_sample", "end_sample", "split",
    "fault_type", "fault_size_in", "outer_race_position", "load_hp",
    "speed_rpm", "sampling_rate_hz", "label", "label4", "label10",
}

# rpm-derived. The team's own domain_shift.csv showed all five sharing an
# identical standardized_shift of 2.208226 -- they restate "different loads run
# at different rpm", they do not measure how the vibration differs.
LEAK_COLS = {
    "shaft_frequency_hz", "expected_bpfi_hz", "expected_bpfo_hz",
    "expected_bsf_hz", "expected_ftf_hz",
}


# --------------------------------------------------------------------- adaptation

def coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Align source second-order statistics to the target's.

    Xs_hat = Xs @ Cs^(-1/2) @ Ct^(1/2)

    Only Xt's covariance is touched -- never its labels -- so this stays
    unsupervised domain adaptation.
    """
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)
    whiten = np.real(sla.fractional_matrix_power(Cs, -0.5))
    recolour = np.real(sla.fractional_matrix_power(Ct, 0.5))
    return Xs @ whiten @ recolour


def _kernel(X: np.ndarray, Y: np.ndarray, kind: str, gamma: float) -> np.ndarray:
    if kind == "linear":
        return X @ Y.T
    if kind == "rbf":
        sq = (np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T)
        return np.exp(-gamma * np.clip(sq, 0, None))
    raise ValueError(kind)


def tca_fit(Xs, Xt, dim=10, mu=1.0, kind="linear", gamma=1.0,
            max_anchors=600, seed=0):
    """Transfer Component Analysis (Pan et al. 2011).

    Finds a projection minimising the Maximum Mean Discrepancy between domains
    while retaining variance.

    The eigenproblem is O(n^3) in the number of samples, which is intractable at
    n=5183. The basis is therefore fitted on a stratified subsample of anchor
    points; every sample is still projected through the resulting transform.
    """
    rng = np.random.default_rng(seed)
    if len(Xs) > max_anchors:
        Xs = Xs[rng.choice(len(Xs), max_anchors, replace=False)]
    if len(Xt) > max_anchors:
        Xt = Xt[rng.choice(len(Xt), max_anchors, replace=False)]

    ns, nt = len(Xs), len(Xt)
    X = np.vstack([Xs, Xt])
    n = ns + nt

    e = np.vstack([np.ones((ns, 1)) / ns, -np.ones((nt, 1)) / nt])
    M = e @ e.T                                   # MMD matrix
    M /= np.linalg.norm(M, "fro") + 1e-12
    H = np.eye(n) - np.ones((n, n)) / n           # centering

    K = _kernel(X, X, kind, gamma)
    A = K @ M @ K + mu * np.eye(n)
    B = K @ H @ K

    w, V = sla.eig(A, B + 1e-9 * np.eye(n))
    idx = np.argsort(np.abs(w))[:dim]             # smallest -> least discrepancy
    W = np.real(V[:, idx])

    def transform(Z):
        return _kernel(Z, X, kind, gamma) @ W

    return transform


def apply_adaptation(method, Xs, Xt, args):
    """Return (Xs_new, Xt_new). Target labels are never used anywhere here."""
    if method == "none":
        return Xs, Xt
    if method == "coral":
        return coral(Xs, Xt), Xt
    if method == "tca":
        f = tca_fit(Xs, Xt, dim=args.tca_dim, mu=args.tca_mu,
                    kind=args.tca_kernel, gamma=args.tca_gamma,
                    max_anchors=args.tca_anchors)
        return f(Xs), f(Xt)
    raise ValueError(method)


# --------------------------------------------------------------------------- data

def load_features(path: str, classes: int):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["label4"] = df["fault_type"]
    df["label10"] = np.where(
        df["fault_type"] == "normal", "normal",
        df["fault_type"] + "_" + df["fault_size_in"].map(lambda v: f"{float(v):.3f}"),
    )
    df["label"] = df["label10"] if classes == 10 else df["label4"]

    feat = [c for c in df.columns if c not in META_COLS]
    dropped = [c for c in feat if c in LEAK_COLS or c.startswith("expected_")]
    feat = [c for c in feat if c not in dropped]
    if dropped:
        print(f"[info] dropped rpm-derived columns: {dropped}")
    print(f"[info] {len(df)} windows, {len(feat)} features, {df.label.nunique()} classes")
    return df, feat


def build_tasks(df, mode):
    loads = sorted(df.load_hp.unique())
    if mode == "pairwise":
        return [(f"{s}hp->{t}hp", [s], t) for s in loads for t in loads if s != t]
    if mode == "loo":
        return [(f"rest->{t}hp", [l for l in loads if l != t], t) for t in loads]
    raise ValueError(f"{mode} has no domain shift to adapt")


def make_models(seed):
    """Identical to baseline.py. Tree models are included on purpose: CORAL helps
    distance-based learners far more, and showing that contrast is informative."""
    return {
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "svm_rbf": SVC(C=10, gamma="scale", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }


# --------------------------------------------------------------------------- run

def run(args):
    df, feat_cols = load_features(args.features, args.classes)
    labels = sorted(df.label.unique())
    tasks = build_tasks(df, args.mode)
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    rows = []
    for name, src, tgt in tasks:
        tr = df[df.load_hp.isin(src)]
        te = df[df.load_hp == tgt]
        Xtr_raw, ytr = tr[feat_cols].to_numpy(float), tr.label.to_numpy()
        Xte_raw, yte = te[feat_cols].to_numpy(float), te.label.to_numpy()

        # Scaler fitted on source only, exactly as in baseline.py. Adaptation
        # happens after scaling so both pipelines share an identical starting point.
        sc = StandardScaler().fit(Xtr_raw)
        Xtr0, Xte0 = sc.transform(Xtr_raw), sc.transform(Xte_raw)

        for method in methods:
            Xtr, Xte = apply_adaptation(method, Xtr0, Xte0, args)
            for seed in seeds:
                for mname, model in make_models(seed).items():
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    pcf = f1_score(yte, pred, average=None, labels=labels, zero_division=0)
                    rows.append({
                        "mode": args.mode, "n_classes": args.classes, "task": name,
                        "source": [int(s) for s in src], "target": int(tgt),
                        "method": mname, "adaptation": method, "seed": seed,
                        "n_train": len(tr), "n_test": len(te),
                        "accuracy": float(accuracy_score(yte, pred)),
                        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
                        "per_class_f1": {c: float(v) for c, v in zip(labels, pcf)},
                        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
                    })
            acc = np.mean([r["accuracy"] for r in rows
                           if r["task"] == name and r["adaptation"] == method])
            print(f"  {name:14s} {method:6s}  mean acc over models/seeds = {acc:.4f}")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{args.classes}class"
    with open(out / f"adapt_raw_{tag}.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    res = pd.DataFrame(rows)
    res.groupby(["task", "adaptation", "method"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
    ).round(4).reset_index().to_csv(out / f"adapt_summary_{tag}.csv", index=False)

    print("\n== accuracy by classifier x adaptation (averaged over tasks/seeds) ==")
    print(res.pivot_table(index="method", columns="adaptation",
                          values="accuracy").round(4).to_string())

    base = res[res.adaptation == "none"]
    if len(base):
        print("\n== gain over the no-adaptation baseline, same classifier ==")
        piv = res.pivot_table(index="method", columns="adaptation", values="accuracy")
        for m in piv.columns:
            if m != "none":
                print(f"\n  {m}:")
                print((piv[m] - piv["none"]).round(4).sort_values(ascending=False).to_string())

        print("\n== per task ==")
        pt = res.pivot_table(index="task", columns="adaptation", values="accuracy").round(4)
        print(pt.to_string())

    print(f"\nsaved -> {out}/adapt_*_{tag}.*")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features.csv")
    p.add_argument("--classes", type=int, default=10, choices=[4, 10])
    p.add_argument("--mode", default="loo", choices=["loo", "pairwise"])
    p.add_argument("--methods", default="none,coral,tca")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--tca-dim", type=int, default=10)
    p.add_argument("--tca-mu", type=float, default=1.0)
    p.add_argument("--tca-kernel", default="linear", choices=["linear", "rbf"])
    p.add_argument("--tca-gamma", type=float, default=1.0)
    p.add_argument("--tca-anchors", type=int, default=600,
                   help="samples used to fit the TCA basis (eig is O(n^3))")
    p.add_argument("--outdir", default="results")
    run(p.parse_args())
