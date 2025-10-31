import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def load_data():
    transaction = pd.read_csv("ieee-fraud-detection/train_transaction.csv")
    identity = pd.read_csv("ieee-fraud-detection/train_identity.csv")
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df

def preprocess_data(df):
    if "isFraud" in df.columns:
        df = df.drop(columns=["isFraud"])

    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        v_df = df[v_cols]
        nan_patterns = v_df.isna().astype(int).T
        groups = []
        used = set()
        for c in v_cols:
            if c in used:
                continue
            pattern = tuple(v_df[c].isna())
            group = [g for g in v_cols if tuple(v_df[g].isna()) == pattern]
            groups.append(group)
            used.update(group)

        reduced = pd.DataFrame(index=df.index)
        for g in groups:
            sub = v_df[g].fillna(v_df[g].median(numeric_only=True))
            if len(g) > 10:
                p = PCA(n_components=1, random_state=42)
                reduced[f"Vgrp_{g[0]}"] = p.fit_transform(sub)
            else:
                reduced[f"Vgrp_{g[0]}"] = sub.mean(axis=1)
        df = pd.concat([df.drop(columns=v_cols), reduced], axis=1)
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("missing")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    X = preprocessor.fit_transform(df)
    return X, preprocessor

def run_pca(n_components=20):
    df = load_data()
    X, preprocessor = preprocess_data(df)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    os.makedirs(os.path.dirname("data/processed/pca_features.csv"), exist_ok=True)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df.to_csv("data/processed/pca_features.csv", index=False)
    
    var = pca.explained_variance_ratio_
    print(f"[INFO] PCA complete: {n_components} components explain {var.sum():.2%} of variance")
    print("[INFO] Saved reduced data to: data/processed/pca_features.csv")
    
    return pca_df, pca, preprocessor

if __name__ == "__main__":
    run_pca(n_components=50)