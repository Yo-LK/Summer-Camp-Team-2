"""
CWRU baseline runner (role 3).

"CORAL/DANN이랑 공정하게 비교할 기준점(target_scarce)이 없었어서 추가했고, 
데이터 불균형 보정(class_weight)도 같이 넣었어. 
기존 3개 모드는 코드 그대로라 이전 결과엔 영향 없어."

[변경 사항]
모드 1개 추가	target_scarce — 소스 도메인 하나도 안 쓰고, 타깃 도메인 자기 자신의 라벨 중 클래스당 10%(조절 가능)만 써서 학습 → 같은 타깃의 나머지로 평가
새 함수 1개	subsample_frac_per_class() — 그 10%를 클래스별로 균등하게 뽑아내는 함수
기존 함수 수정	make_models()에 class_weight="balanced" 추가 (outer_race 클래스가 52개 fault 중 28개=54%라 불균형 보정)
run() 로직 수정	target_scarce일 때만 위 서브샘플링을 거치도록 분기 추가
CLI 옵션 추가	--target-frac (기본 0.10)

Produces the three reference numbers every transfer-learning result is judged against:
  1. same-domain upper bound   : train/test inside one load, no domain shift
  2. cross-domain lower bound  : train on source load(s), test on target load, NO adaptation
  3. per-class breakdown       : ball / inner / outer separately (ball is the hard case)

Usage
-----
  python baseline.py --features features.csv --classes 10 --mode loo
  python baseline.py --features features.csv --classes 4  --mode pairwise
  python baseline.py --features features.csv --mode same_domain

Output
------
  results/baseline_raw.json     one record per (task, model, seed)
  results/baseline_summary.csv  mean +/- std pivot table
"""
from __future__ import annotations

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

META_COLS = {
    "record_id", "window_number", "start_sample", "end_sample", "split",
    "fault_type", "fault_size_in", "outer_race_position", "load_hp",
    "speed_rpm", "sampling_rate_hz", "label", "label4", "label10",
}


# --------------------------------------------------------------------------- data

def load_features(path: str, classes: int) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    df["label4"] = df["fault_type"]
    df["label10"] = np.where(
        df["fault_type"] == "normal",
        "normal",
        df["fault_type"] + "_" + df["fault_size_in"].map(lambda v: f"{float(v):.3f}"),
    )
    df["label"] = df["label10"] if classes == 10 else df["label4"]

    feat_cols = [c for c in df.columns if c not in META_COLS]
    # rpm-derived constants leak the operating condition itself -> drop them.
    # keeping them lets a tree memorise "load 3 == rpm 1730" instead of learning the fault.
    leak = [c for c in feat_cols if c.startswith("expected_") or c == "shaft_frequency_hz"]
    feat_cols = [c for c in feat_cols if c not in leak]
    if leak:
        print(f"[info] dropped condition-leaking columns: {leak}")
    print(f"[info] {len(df)} windows, {len(feat_cols)} features, {df.label.nunique()} classes")
    return df, feat_cols


# --------------------------------------------------------------------------- splits

def split_same_domain(df: pd.DataFrame, load: int, holdout: float = 0.3):
    """Within one load. Per record: first 70% of windows train, last 30% test.

    One window is dropped at the boundary because consecutive windows overlap 50%;
    without the gap the same samples appear on both sides and accuracy is inflated.
    """
    tr, te = [], []
    for _, g in df[df.load_hp == load].groupby("record_id"):
        g = g.sort_values("start_sample")
        cut = int(len(g) * (1 - holdout))
        tr.append(g.iloc[: max(cut - 1, 1)])
        te.append(g.iloc[cut:])
    return pd.concat(tr), pd.concat(te)


def split_cross_domain(df: pd.DataFrame, src_loads: list[int], tgt_load: int):
    return df[df.load_hp.isin(src_loads)], df[df.load_hp == tgt_load]


def subsample_frac_per_class(df: pd.DataFrame, frac: float, seed: int, min_per_class: int = 1) -> pd.DataFrame:
    """클래스별로 frac 비율만 남김 (target_scarce용). groupby().apply()는 최신 pandas에서
    그룹 기준 컬럼(label)을 결과에서 빼버리는 버그가 있어서 .groups+.loc 방식으로 짬."""
    rng = np.random.RandomState(seed)
    keep_idx = []
    for _, idx in df.groupby("label").groups.items():
        idx = np.asarray(idx)
        n_keep = min(len(idx), max(min_per_class, round(len(idx) * frac)))
        keep_idx.extend(rng.choice(idx, size=n_keep, replace=False))
    return df.loc[keep_idx]

def build_tasks(df: pd.DataFrame, mode: str):
    loads = sorted(df.load_hp.unique())
    if mode == "pairwise":
        return [(f"{s}hp->{t}hp", [s], t) for s in loads for t in loads if s != t]
    if mode == "loo":
        return [(f"rest->{t}hp", [l for l in loads if l != t], t) for t in loads]
    if mode == "same_domain":
        return [(f"{l}hp(in-domain)", [l], l) for l in loads]
    if mode == "target_scarce":                     # ← 이 2줄 추가
        return [(f"{l}hp(target_scarce)", [], l) for l in loads]
    raise ValueError(mode)


# --------------------------------------------------------------------------- models

def make_models(seed: int) -> dict:
    m = {
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "logreg": LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),   # 추가
        "svm_rbf": SVC(C=10, gamma="scale", random_state=seed, class_weight="balanced"),            # 추가
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced"),  # 추가
        "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced"),      # 추가
        "grad_boost": GradientBoostingClassifier(random_state=seed),   # class_weight 파라미터 없음, 그대로 둠
    }
    try:
        from xgboost import XGBClassifier      # optional
        m["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass
    return m


def fit_eval(model, Xtr, ytr, Xte, yte, labels):
    """Scaler is fitted on SOURCE ONLY. Fitting on target = leaking the target
    distribution, which is itself a (weak) form of adaptation and voids the baseline."""
    needs_xgb_int = model.__class__.__name__ == "XGBClassifier"
    lut = {c: i for i, c in enumerate(labels)}
    if needs_xgb_int:
        ytr, yte = np.array([lut[v] for v in ytr]), np.array([lut[v] for v in yte])

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    inv = {i: c for c, i in lut.items()}
    if needs_xgb_int:
        yte = np.array([inv[v] for v in yte])
        pred = np.array([inv[v] for v in pred])

    per_class = f1_score(yte, pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "per_class_f1": {c: float(v) for c, v in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------- run

def run(args):
    df, feat_cols = load_features(args.features, args.classes)
    labels = sorted(df.label.unique())
    tasks = build_tasks(df, args.mode)
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for name, src, tgt in tasks:
        if args.mode == "same_domain":
            tr, te = split_same_domain(df, tgt)
        elif args.mode == "target_scarce":                       # ← 이 분기 추가
            tr_full, te = split_same_domain(df, tgt)
        else:
            tr, te = split_cross_domain(df, src, tgt)

        if args.mode != "target_scarce":                         # ← 이 조건 추가
            Xtr, ytr = tr[feat_cols].to_numpy(), tr.label.to_numpy()
        Xte, yte = te[feat_cols].to_numpy(), te.label.to_numpy()

        for seed in seeds:
            if args.mode == "target_scarce":                     # ← 이 블록 추가
                tr = subsample_frac_per_class(tr_full, args.target_frac, seed=seed)
                Xtr, ytr = tr[feat_cols].to_numpy(), tr.label.to_numpy()

            for mname, model in make_models(seed).items():
                r = fit_eval(model, Xtr, ytr, Xte, yte, labels)
                rows.append({
                    "mode": args.mode, "n_classes": args.classes, "task": name,
                    "source": [int(s) for s in src], "target": int(tgt),
                    "method": mname, "seed": seed,
                    "adaptation": "none", "n_train": len(tr), "n_test": len(te), **r,
                })
                print(f"  {name:16s} {mname:15s} seed={seed}  "
                      f"acc={r['accuracy']:.4f}  macroF1={r['f1_macro']:.4f}")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{args.classes}class"
    with open(out / f"baseline_raw_{tag}.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    res = pd.DataFrame(rows)
    summary = (res.groupby(["task", "method"])
                  .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                       f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
                  .round(4).reset_index())
    summary.to_csv(out / f"baseline_summary_{tag}.csv", index=False)

    print("\n== accuracy (mean over seeds) ==")
    print(res.pivot_table(index="method", columns="task", values="accuracy").round(4).to_string())

    pc = pd.DataFrame([r["per_class_f1"] for r in rows])
    print("\n== per-class F1, averaged over all tasks/models/seeds ==")
    print(pc.mean().round(4).sort_values().to_string())
    print(f"\nsaved -> {out}/baseline_*_{tag}.*")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features.csv")
    p.add_argument("--classes", type=int, default=10, choices=[4, 10])
    p.add_argument("--mode", default="loo", choices=["loo", "pairwise", "same_domain", "target_scarce"])
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--outdir", default="results")
    p.add_argument("--target-frac", type=float, default=0.10,      # ← 새 옵션 추가
                    help="target_scarce only: per-class fraction of target load's train windows to keep")
    run(p.parse_args())
