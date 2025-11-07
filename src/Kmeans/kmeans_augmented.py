#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def run(
    csv,
    k,
    seed,
    test_size,
    fit_subsample,
    metric_subsample,
    outdir,
):
    csv_path = Path(csv)
    df = pd.read_csv(csv_path, low_memory=True)

    y = df["isFraud"].astype(int).values
    X = df.drop(columns=["isFraud"]).select_dtypes(include=[np.number]).copy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_fit = min(fit_subsample, X_train_scaled.shape[0])
    fit_idx = np.random.default_rng(seed).choice(X_train_scaled.shape[0], size=n_fit, replace=False)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(X_train_scaled[fit_idx])

    labels_train = kmeans.predict(X_train_scaled)
    labels_test = kmeans.predict(X_test_scaled)

    mapping = {}
    pred_train = np.zeros_like(labels_train)
    for c in np.unique(labels_train):
        mask = labels_train == c
        mapping[c] = 1 if y_train[mask].mean() >= 0.5 else 0
        pred_train[mask] = mapping[c]
    pred_test = np.array([mapping[c] for c in labels_test])

    n_metric = min(metric_subsample, X_train_scaled.shape[0])
    metric_idx = np.random.default_rng(seed).choice(X_train_scaled.shape[0], size=n_metric, replace=False)

    ari = float(adjusted_rand_score(y_train[metric_idx], labels_train[metric_idx]))
    nmi = float(normalized_mutual_info_score(y_train[metric_idx], labels_train[metric_idx]))
    h = float(homogeneity_score(y_train[metric_idx], labels_train[metric_idx]))
    c = float(completeness_score(y_train[metric_idx], labels_train[metric_idx]))
    v = float(v_measure_score(y_train[metric_idx], labels_train[metric_idx]))

    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )
    beta2 = 4
    f2 = (1 + beta2) * (precision * recall) / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_out = df.iloc[train_idx].copy()
    df_test_out = df.iloc[test_idx].copy()

    df_train_out["kmeans_cluster"] = labels_train
    df_test_out["kmeans_cluster"] = labels_test

    df_train_out["kmeans_fraud_pred"] = pred_train
    df_test_out["kmeans_fraud_pred"] = pred_test

    out_train_path = out_dir / "kmeans_on_augmented_train_split.csv"
    out_test_path = out_dir / "kmeans_on_augmented_test_split.csv"

    df_train_out.to_csv(out_train_path, index=False)
    df_test_out.to_csv(out_test_path, index=False)

    print("CSV:", csv_path)
    print("k:", k, " seed:", seed, " mapping:", mapping)
    print("External(train)  ARI", f"{{ari:.4f}}", "| NMI", f"{{nmi:.4f}}", "| H", f"{{h:.4f}}", "| C", f"{{c:.4f}}", "| V", f"{{v:.4f}}")
    print("Test  TN", tn, " FP", fp, " FN", fn, " TP", tp, "| Precision", f"{{precision:.4f}}", " Recall", f"{{recall:.4f}}", " F1", f"{{f1:.4f}}", " F2", f"{{f2:.4f}}")
    print("Wrote:", out_train_path)
    print("Wrote:", out_test_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/mnt/data/ieee-fraud-detection-2/ieee-fraud-detection-2/ieee-fraud-detection/augmented_processed.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--fit-subsample", type=int, default=3000)
    parser.add_argument("--metric-subsample", type=int, default=3000)
    parser.add_argument("--outdir", type=str, default="./results/unsupervised")

    args = parser.parse_args()

    run(
        csv=args.csv,
        k=args.k,
        seed=args.seed,
        test_size=args.test_size,
        fit_subsample=args.fit_subsample,
        metric_subsample=args.metric_subsample,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
