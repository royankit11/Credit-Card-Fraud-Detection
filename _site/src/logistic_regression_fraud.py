import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse
import joblib
import os

SAMPLE_SIZE = None  # use smaller sample for testing, set to None for full dataset

# ---------- Load and Merge Data ----------
def load_data():
    transaction = pd.read_csv("train_transaction.csv")
    identity = pd.read_csv("train_identity.csv")
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df

def load_data_optimized():
    print("[INFO] Loading data...")
    transaction = pd.read_csv("train_transaction.csv", usecols=lambda c: c != "TransactionDT")
    identity = pd.read_csv("train_identity.csv")
    df = transaction.merge(identity, on="TransactionID", how="left")
    if SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
        print(f"[INFO] Using sample of {len(df):,} rows")
    return df

# ---------- Preprocess Original Features ----------
def preprocess_data(df):
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[categorical_cols] = X[categorical_cols].fillna("missing")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    return X, y, preprocessor

def preprocess_data_optimized(df):
    print("[INFO] Preprocessing data...")
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Fill missing values
    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[categorical_cols] = X[categorical_cols].fillna("missing")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    print(f"[INFO] Transformed shape: {X_transformed.shape}")
    if not sparse.issparse(X_transformed):
        X_transformed = sparse.csr_matrix(X_transformed)

    return X_transformed, y, preprocessor


# ---------- Train Logistic Regression ----------
def train_logistic_regression(X, y, model_name="original"):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))

    clf = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        class_weight=class_weights,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    print(f"\n[RESULTS] Logistic Regression on {model_name}")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, f"models/logreg_{model_name}.pkl")

def train_logistic_regression_optimized(X, y, model_name="original"):
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Computing class weights...")
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))

    print("[INFO] Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=300,
        solver="saga",
        class_weight=class_weights,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n[RESULTS] Logistic Regression ({model_name})")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, f"models/logreg_{model_name}.pkl")
    print(f"[SAVED] Model saved as models/logreg_{model_name}.pkl")

# ---------- Run Logistic Regression on PCA features ----------
def run_pca_logreg():
    # Load PCA data
    pca_df = pd.read_csv("pca_features.csv")
    original = load_data()
    y = original["isFraud"].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        pca_df, y, test_size=0.2, random_state=42, stratify=y
    )

    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))

    clf = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        class_weight=class_weights,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n[RESULTS] Logistic Regression on PCA features")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "models/logreg_pca.pkl")

# ---------- Main ----------
if __name__ == "__main__":
    print("[INFO] Loading and preprocessing original dataset...")
    df = load_data_optimized()
    X, y, preprocessor = preprocess_data_optimized(df)

    print("[INFO] Training logistic regression on original features...")
    train_logistic_regression_optimized(X, y, model_name="lightweight")


    # df = load_data()
    # X, y, preprocessor = preprocess_data(df)

    # print("[INFO] Fitting preprocessing pipeline...")
    # X_transformed = preprocessor.fit_transform(X)

    # print("[INFO] Training logistic regression on original features...")
    # train_logistic_regression(X, y, model_name="normal")

    print("[INFO] Training logistic regression on PCA features...")
    run_pca_logreg()
