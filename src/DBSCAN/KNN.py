import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==== CONFIG ====
TRAIN_FILE = "train.csv"
MIN_PTS_LIST = [2, 4, 7, 10, 15]
YLIM_MAX = 15
# =================


def load_features_only(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore").fillna(0.0)
    return X


def main():
    X_tr = load_features_only(TRAIN_FILE)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)

    results = []

    plt.figure(figsize=(8, 5))

    for min_pts in MIN_PTS_LIST:
        # k = MinPts in k-distance plot
        neigh = NearestNeighbors(n_neighbors=min_pts)
        neigh.fit(X_scaled)
        distances, _ = neigh.kneighbors(X_scaled)

        # distance to k-th neighbor, sorted
        k_dist = distances[:, -1]
        k_dist_sorted = np.sort(k_dist)

        # plot curve
        plt.plot(
            np.arange(len(k_dist_sorted)),
            k_dist_sorted,
            label=f"MinPts={min_pts}"
        )

        # simple "elbow" heuristics: upper percentiles
        eps90 = np.percentile(k_dist_sorted, 90)
        eps95 = np.percentile(k_dist_sorted, 95)
        eps99 = np.percentile(k_dist_sorted, 99)

        results.append({
            "min_pts": min_pts,
            "eps90": eps90,
            "eps95": eps95,
            "eps99": eps99,
        })

    # plot cosmetics
    plt.xlabel("Points sorted by distance")
    plt.ylabel("k-distance")
    plt.title("k-distance curves for multiple MinPts")
    plt.ylim(0, YLIM_MAX)        # limit y-axis as you asked
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("knn_eps_multi.png", dpi=300)
    plt.show()

    print("\nSuggested eps values from k-distance percentiles:")
    print(f"{'MinPts':>7} {'eps90':>10} {'eps95':>10} {'eps99':>10}")
    for r in results:
        print(f"{r['min_pts']:7d} {r['eps90']:10.4f} {r['eps95']:10.4f} {r['eps99']:10.4f}")

    print("\nUse one of these eps values (e.g. eps95) for DBSCAN with the same MinPts.")


if __name__ == "__main__":
    main()