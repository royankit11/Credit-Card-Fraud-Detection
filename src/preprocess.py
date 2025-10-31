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
    run_pca(n_components=20)