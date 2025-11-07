# kmeans_split_eval.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, pairwise_distances,
    confusion_matrix, precision_recall_fscore_support, jaccard_score,
    fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score,
    v_measure_score
)
try:
    from sklearn.metrics import rand_score
    HAVE_RAND = True
except Exception:
    HAVE_RAND = False

# ------------------ CONFIG ------------------
FILE = "augmented_processed.csv"
K = 2
SEED = 10
TEST_SIZE = 0.25
SIL_SAMPLE = 20000
BETACV_SAMPLE = 5000
# --------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' not found.")
    y = df["isFraud"].astype(int).to_numpy()
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").fillna(0.0)
    return X, y

def standardize_train_test(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)

def run_kmeans_train_predict(Xtr, Xte, k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=300, algorithm="elkan")
    ytr_k = km.fit_predict(Xtr)
    yte_k = km.predict(Xte)
    return ytr_k, yte_k

def safe_silhouette(X, labels, sample_size):
    try:
        return float(silhouette_score(X, labels, sample_size=min(sample_size, len(X))))
    except Exception:
        return np.nan

def beta_cv(X, labels, sample_size, seed):
    n = len(X)
    if n < 3: return np.nan
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    Xs, ls = X[idx], np.asarray(labels)[idx]
    D = pairwise_distances(Xs)
    iu = np.triu_indices_from(D, k=1)
    same = (ls[:, None] == ls[None, :])
    intra = D[iu][same[iu]]
    inter = D[iu][~same[iu]]
    if len(intra) == 0 or len(inter) == 0: return np.nan
    return float(intra.mean() / inter.mean())

def two_by_k_table(y_true_bin, clusters, k):
    # rows: actual 0/1; cols: cluster 0..k-1
    tab = np.zeros((2, k), dtype=int)
    for c in range(k):
        mask = (clusters == c)
        tab[0, c] = int(((y_true_bin == 0) & mask).sum())
        tab[1, c] = int(((y_true_bin == 1) & mask).sum())
    return tab

def learn_label_map_k(y_true_bin, clusters, k):
    """Return set of cluster ids to label as 'fraud'=1."""
    tab = two_by_k_table(y_true_bin, clusters, k)
    totals = tab.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_rate = np.where(totals > 0, tab[1] / totals, 0.0)
    prevalence = y_true_bin.mean()

    # primary rule: fraud if cluster pos_rate > prevalence
    pos_clusters = [c for c in range(k) if pos_rate[c] > prevalence]

    # guardrails to avoid 'all 1s' or 'all 0s'
    if len(pos_clusters) == 0:
        pos_clusters = [int(np.argmax(pos_rate))]         
    elif len(pos_clusters) == k:
        pos_clusters = [int(np.argmax(pos_rate))]          

    # diagnostics
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
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average="binary", zero_division=0)
    jacc = jaccard_score(y_true, y_pred_bin, average="binary", zero_division=0)
    fm = fowlkes_mallows_score(y_true, y_pred_bin)
    ari = adjusted_rand_score(y_true, y_pred_bin)
    vme = v_measure_score(y_true, y_pred_bin)
    nmi = normalized_mutual_info_score(y_true, y_pred_bin)
    rnd = rand_score(y_true, y_pred_bin) if HAVE_RAND else np.nan
    return cm, prec, rec, f1, jacc, rnd, ari, fm, vme, nmi

def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.show()

def main():
   
    X, y = load_data(FILE)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    
    Xtr, Xte = standardize_train_test(X_tr, X_te)

   
    ytr_k, yte_k = run_kmeans_train_predict(Xtr, Xte, K, SEED)

    
    for split, Xs, ls in [("Train", Xtr, ytr_k), ("Test", Xte, yte_k)]:
        sil = safe_silhouette(Xs, ls, SIL_SAMPLE)
        dbi = davies_bouldin_score(Xs, ls)
        bcv = beta_cv(Xs, ls, BETACV_SAMPLE, SEED)
        print(f"\n[{split}] Internal")
        print(f"  Silhouette: {sil:.4f}  |  Davies–Bouldin: {dbi:.4f}  |  BetaCV: {bcv:.4f}")

    
    print("\n[Train] 2×k confusion (rows=actual 0/1, cols=cluster id):")
    print(pd.DataFrame(two_by_k_table(y_tr, ytr_k, K), index=["y=0", "y=1"]))

    print("\n[Test]  2×k confusion (rows=actual 0/1, cols=cluster id):")
    print(pd.DataFrame(two_by_k_table(y_te, yte_k, K), index=["y=0", "y=1"]))

    
    pos_clusters, prev, diag = learn_label_map_k(y_tr, ytr_k, K)
    print("\n[Mapping learned on TRAIN]")
    print(diag.to_string(index=False))
    print(f"Global train prevalence: {prev:.4f}")
    print(f"Clusters marked as FRAUD (1): {sorted(pos_clusters)}")

    
    ytr_pred = binarize_by_map(ytr_k, pos_clusters)
    yte_pred = binarize_by_map(yte_k, pos_clusters)

    
    for split, y_true, y_pred in [("Train", y_tr, ytr_pred), ("Test", y_te, yte_pred)]:
        cm, prec, rec, f1, jacc, rnd, ari, fm, vme, nmi = external_metrics(y_true, y_pred)
        print(f"\n[{split}] External (vs isFraud)")
        print("Confusion matrix [rows=actual 0/1, cols=pred 0/1]:")
        print(cm)
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"Jaccard: {jacc:.4f} | Rand: {rnd:.4f} | Adjusted Rand: {ari:.4f} | Fowlkes–Mallows: {fm:.4f}")
        print(f"V-measure: {vme:.4f} | NMI: {nmi:.4f}")

    
    plot_confusion(confusion_matrix(y_tr, ytr_pred, labels=[0,1]), "Train Confusion Matrix (Mapped)")
    plot_confusion(confusion_matrix(y_te, yte_pred, labels=[0,1]), "Test Confusion Matrix (Mapped)")

if __name__ == "__main__":
    main()

