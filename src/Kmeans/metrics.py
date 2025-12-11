import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

def pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)

def roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def threshold_table(y_true, y_score, thresholds=(0.3, 0.5, 0.7)):
    out = []
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = f1_score(y_true, y_pred)
        out.append({"threshold": float(th), "precision": float(prec), "recall": float(rec), "f1": float(f1)})
    return out

def silhouette(X, labels):
    if len(np.unique(labels)) <= 1:
        return float("nan")
    return float(silhouette_score(X, labels))

def beta_cv(X, labels):
    D = pairwise_distances(X)
    labs = np.asarray(labels)
    clusters = np.unique(labs)
    intra = []
    for k in clusters:
        idx = np.where(labs == k)[0]
        if len(idx) >= 2:
            d = D[np.ix_(idx, idx)]
            vals = d[np.triu_indices_from(d, 1)]
            if vals.size > 0:
                intra.append(vals.mean())
    inter = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            i1 = np.where(labs == clusters[i])[0]
            i2 = np.where(labs == clusters[j])[0]
            if len(i1) > 0 and len(i2) > 0:
                inter.append(D[np.ix_(i1, i2)].mean())
    intra_m = np.nan if len(intra) == 0 else float(np.mean(intra))
    inter_m = np.nan if len(inter) == 0 else float(np.mean(inter))
    if np.isnan(intra_m) or np.isnan(inter_m):
        return float("nan")
    return float(intra_m / (inter_m + 1e-9))
