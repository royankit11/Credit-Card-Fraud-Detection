import json, os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.Kmeans.metrics import pr_auc, roc_auc, threshold_table, silhouette, beta_cv

def fit_gmm_scores(X, y=None, ks=(2, 3, 4), cov_types=("full", "diag"), use_pca=False, pca_var=0.95, random_state=42):
    Xs = StandardScaler().fit_transform(X)
    if use_pca:
        Xs = PCA(n_components=pca_var, svd_solver="full", random_state=random_state).fit_transform(Xs)
    best = None
    best_bic = np.inf
    for k in ks:
        for cov in cov_types:
            gm = GaussianMixture(n_components=k, covariance_type=cov, n_init=5, max_iter=500, reg_covar=1e-5, init_params="kmeans", random_state=random_state)
            gm.fit(Xs)
            bic = gm.bic(Xs)
            if bic < best_bic:
                best_bic = bic
                best = gm
    resp = best.predict_proba(Xs)
    labels = resp.argmax(axis=1)
    if y is not None:
        mass = resp.sum(axis=0) + 1e-9
        comp_fraud = (resp.T @ y) / mass
        cutoff = np.median(comp_fraud)
        fraud_like = comp_fraud >= cutoff
        score = resp[:, fraud_like].sum(axis=1)
    else:
        score = resp.max(axis=1)
    return {"model": best, "resp": resp, "labels": labels, "score": score, "Xs": Xs}

def run(input_csv, label_col, out_dir, feature_cols=None, use_pca=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].select_dtypes(include=[np.number]).to_numpy()
    y = df[label_col].to_numpy().astype(int)
    res = fit_gmm_scores(X, y, use_pca=use_pca)
    score = res["score"]
    labels = res["labels"]
    Xs = res["Xs"]
    metrics = {
        "roc_auc": float(roc_auc(y, score)),
        "pr_auc": float(pr_auc(y, score)),
        "thresholds": threshold_table(y, score, thresholds=(0.3, 0.5, 0.7)),
        "silhouette": float(silhouette(Xs, labels)),
        "beta_cv": float(beta_cv(Xs, labels))
    }
    with open(os.path.join(out_dir, "gmm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"score": score, label_col: y}).to_csv(os.path.join(out_dir, "gmm_scores.csv"), index=False)
    return metrics
