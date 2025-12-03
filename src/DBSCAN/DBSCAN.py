import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, pairwise_distances,
    confusion_matrix, precision_recall_fscore_support, jaccard_score,
    fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score,
    v_measure_score
)

# ------------------ CONFIG ------------------
TRAIN_FILE = "train.csv"       # your train CSV
TEST_FILE  = "test.csv"        # your test CSV

# Grid of hyperparameters
EPS_GRID = [2.4]
MIN_SAMPLES_GRID = [2]

SIL_SAMPLE = 20000
BETACV_SAMPLE = 5000
SEED = 10

try:
    from sklearn.metrics import rand_score
    HAVE_RAND = True
except ImportError:
    HAVE_RAND = False
# --------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' not found in " + path)
    y = df["isFraud"].astype(int).to_numpy()
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").fillna(0.0)
    return X, y

def standardize_train_test(X_train, X_test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    return Xtr, Xte

def safe_silhouette(X, labels, sample_size):
    try:
        return float(
            silhouette_score(X, labels,
                             sample_size=min(sample_size, len(X)))
        )
    except Exception:
        return np.nan

def beta_cv(X, labels, sample_size, seed):
    n = len(X)
    if n < 3:
        return np.nan
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    Xs = X[idx]
    ls = np.asarray(labels)[idx]

    D = pairwise_distances(Xs)
    iu = np.triu_indices_from(D, k=1)

    same = (ls[:, None] == ls[None, :])
    intra = D[iu][same[iu]]
    inter = D[iu][~same[iu]]

    if len(intra) == 0 or len(inter) == 0:
        return np.nan

    return float(intra.mean() / inter.mean())

# ---------- label mapping utilities

def two_by_k_table(y_true_bin, clusters, k):
    """rows: actual 0/1; cols: cluster 0..k-1"""
    tab = np.zeros((2, k), dtype=int)
    for c in range(k):
        mask = (clusters == c)
        tab[0, c] = int(((y_true_bin == 0) & mask).sum())
        tab[1, c] = int(((y_true_bin == 1) & mask).sum())
    return tab

def learn_label_map_k(y_true_bin, clusters, k):
    """
    Return set of cluster ids to label as 'fraud'=1,
    using prevalence comparison like in your KMeans code.
    """
    tab = two_by_k_table(y_true_bin, clusters, k)
    totals = tab.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_rate = np.where(totals > 0, tab[1] / totals, 0.0)
    prevalence = y_true_bin.mean()

    # label as positive if cluster's positive rate > prevalence
    pos_clusters = [c for c in range(k) if pos_rate[c] > prevalence]

    # avoid degenerate cases
    if len(pos_clusters) == 0:
        pos_clusters = [int(np.argmax(pos_rate))]
    elif len(pos_clusters) == k:
        pos_clusters = [int(np.argmax(pos_rate))]

    diag = pd.DataFrame({
        "cluster": np.arange(k),
        "n_total": totals,
        "n_pos": tab[1],
        "n_neg": tab[0],
        "pos_rate": pos_rate
    }).sort_values("pos_rate", ascending=False)

    return set(pos_clusters), prevalence, diag

def binarize_by_map(cluster_labels, pos_clusters):
    return np.isin(cluster_labels, list(pos_clusters)).astype(int)

def external_metrics(y_true, y_pred_bin):
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average="binary", zero_division=0
    )
    jacc = jaccard_score(y_true, y_pred_bin, average="binary", zero_division=0)
    fm = fowlkes_mallows_score(y_true, y_pred_bin)
    ari = adjusted_rand_score(y_true, y_pred_bin)
    vme = v_measure_score(y_true, y_pred_bin)
    nmi = normalized_mutual_info_score(y_true, y_pred_bin)
    rnd = rand_score(y_true, y_pred_bin) if HAVE_RAND else np.nan
    return cm, prec, rec, f1, jacc, rnd, ari, fm, vme, nmi

# ---------- DBSCAN helpers ----------

def run_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    return labels

def relabel_to_0_k(labels):
    """
    Map arbitrary cluster labels (including -1) to 0..k-1.
    Returns new_labels, mapping dict.
    """
    unique = np.unique(labels)
    label_to_new = {lab: i for i, lab in enumerate(unique)}
    new_labels = np.vectorize(label_to_new.get)(labels).astype(int)
    return new_labels, label_to_new

def noise_stats(labels):
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    pct_noise = 100.0 * n_noise / max(n_total, 1)
    return n_noise, n_total, pct_noise

def cluster_and_evaluate(X, y, eps, min_samples, sample_seed=SEED):
    """
    Run DBSCAN, map clusters -> {0,1}, compute accuracy + metrics.
    """
    raw_labels = run_dbscan(X, eps, min_samples)
    n_noise, n_total, pct_noise = noise_stats(raw_labels)

    # relabel for mapping (including noise as one of the clusters)
    labels_reindexed, mapping = relabel_to_0_k(raw_labels)
    k = len(np.unique(labels_reindexed))

    pos_clusters, prevalence, diag = learn_label_map_k(y, labels_reindexed, k)
    y_pred = binarize_by_map(labels_reindexed, pos_clusters)

    cm, prec, rec, f1, jacc, rnd, ari, fm, vme, nmi = external_metrics(y, y_pred)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

    # internal metrics
    sil = safe_silhouette(X, labels_reindexed, SIL_SAMPLE)
    try:
        dbi = davies_bouldin_score(X, labels_reindexed)
    except Exception:
        dbi = np.nan
    bcv = beta_cv(X, labels_reindexed, BETACV_SAMPLE, sample_seed)

    metrics = {
        "cm": cm,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "jacc": jacc,
        "rnd": rnd,
        "ari": ari,
        "fm": fm,
        "vme": vme,
        "nmi": nmi,
        "acc": acc,
        "sil": sil,
        "dbi": dbi,
        "bcv": bcv,
        "n_noise": n_noise,
        "n_total": n_total,
        "pct_noise": pct_noise,
        "pos_clusters": pos_clusters,
        "prevalence": prevalence,
        "diag": diag,
        "mapping": mapping,
    }
    return raw_labels, labels_reindexed, y_pred, metrics


def plot_confusion_matrix(cm, split_name):
    """
    Plot 2x2 confusion matrix as percentages of all samples.
    cm: raw count matrix (2x2)
    """
    cm = np.asarray(cm)
    total = cm.sum()
    cm_pct = cm.astype(float) / max(total, 1) * 100.0

    plt.figure(figsize=(4, 4))
    plt.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    plt.title(f"{split_name} confusion matrix (%)")
    plt.colorbar(label="% of samples")

    tick_marks = np.arange(2)
    classes = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center", va="center", color="black"
            )

    plt.tight_layout()
    plt.show()



# ---------- main ----------

def main():
    X_tr, y_tr = load_data(TRAIN_FILE)
    X_te, y_te = load_data(TEST_FILE)

    Xtr, Xte = standardize_train_test(X_tr, X_te)

    # 3) Hyperparameter search on TRAIN ONLY, using accuracy
    best = {
        "eps": None,
        "min_samples": None,
        "acc": -1.0,
        "metrics": None,
        "raw_labels": None,
        "labels_reindexed": None,
        "y_pred": None,
    }

    print("=== Hyperparameter search (train only) ===")
    for eps in EPS_GRID:
        for ms in MIN_SAMPLES_GRID:
            raw_labels, labels_reindexed, y_pred, metrics = cluster_and_evaluate(
                Xtr, y_tr, eps, ms
            )
            acc = metrics["acc"]
            print(f"eps={eps}, min_samples={ms} -> train acc={acc:.4f}, F1={metrics['f1']:.4f}, noise={metrics['pct_noise']:.2f}%")

            if acc > best["acc"]:
                best.update({
                    "eps": eps,
                    "min_samples": ms,
                    "acc": acc,
                    "metrics": metrics,
                    "raw_labels": raw_labels,
                    "labels_reindexed": labels_reindexed,
                    "y_pred": y_pred,
                })

    print("\n=== Best hyperparameters (train) ===")
    print(f"Best eps = {best['eps']}")
    print(f"Best min_samples = {best['min_samples']}")
    print(f"Best train accuracy = {best['acc']:.4f}")
    print(f"Best train F1 = {best['metrics']['f1']:.4f}")
    print(f"Train noise = {best['metrics']['n_noise']}/{best['metrics']['n_total']} ({best['metrics']['pct_noise']:.2f}%)")

    # 4) Final evaluation on Train and Test using best params
    print("\n=== Final evaluation with best params ===")

    # Re-evaluate on train (for clean printout)
    _, _, ytr_pred, m_tr = cluster_and_evaluate(Xtr, y_tr, best["eps"], best["min_samples"])
    # Evaluate on test
    _, _, yte_pred, m_te = cluster_and_evaluate(Xte, y_te, best["eps"], best["min_samples"])

    for split, m in [("Train", m_tr), ("Test", m_te)]:
        print(f"\n[{split}] Internal metrics")
        print(f"  Silhouette:      {m['sil']:.4f}")
        print(f"  Davies–Bouldin:  {m['dbi']:.4f}")
        print(f"  BetaCV:          {m['bcv']:.4f}")

        print(f"\n[{split}] External metrics (vs isFraud)")
        print("  Confusion matrix [rows=actual 0/1, cols=pred 0/1]:")
        print(m["cm"])
        print(f"  Accuracy:  {m['acc']:.4f}")
        print(f"  Precision: {m['prec']:.4f} | Recall: {m['rec']:.4f} | F1: {m['f1']:.4f}")
        print(f"  Jaccard:   {m['jacc']:.4f} | Rand: {m['rnd']:.4f}")
        print(f"  ARI:       {m['ari']:.4f} | FM: {m['fm']:.4f}")
        print(f"  V-measure: {m['vme']:.4f} | NMI: {m['nmi']:.4f}")

        print(f"\n[{split}] Noise stats")
        print(f"  Noise points: {m['n_noise']}/{m['n_total']}  ({m['pct_noise']:.2f}%)")

        print(f"\n[{split}] External metrics (vs isFraud)")
        print("  Confusion matrix (% of all samples, rows=actual 0/1, cols=pred 0/1):")

        cm = m["cm"]
        cm_pct = cm.astype(float) / cm.sum() * 100.0
        print(np.round(cm_pct, 2))  # numeric percentages

        plot_confusion_matrix(cm, split)  # plotted percentages


if __name__ == "__main__":
    main()
