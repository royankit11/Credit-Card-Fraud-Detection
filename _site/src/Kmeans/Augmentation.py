
# Make the minority fraud class (isFraud=1) 1:1 with the majority class
import pandas as pd

IN_PATH = "processed_train.csv"
OUT_PATH = "augmented_processed.csv"
LABEL = "isFraud"
SEED = 42

def main():
    df = pd.read_csv(IN_PATH)
    if LABEL not in df.columns:
        raise ValueError(f"Column '{LABEL}' not found in {IN_PATH}")

    counts = df[LABEL].value_counts()
    if counts.size < 2:
        raise ValueError(f"Need two classes in '{LABEL}' to oversample. Found: {counts.to_dict()}")

    # Identify majority/minority classes
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_majority = counts.max()
    n_minority = counts.min()

    
    if n_majority == n_minority:
        df.to_csv(OUT_PATH, index=False)
        print(f"Already balanced: {counts.to_dict()}. Saved copy to {OUT_PATH}.")
        return

    df_major = df[df[LABEL] == majority_class]
    df_minor = df[df[LABEL] == minority_class]

    
    df_minor_over = df_minor.sample(n=n_majority, replace=True, random_state=SEED)

    
    df_balanced = pd.concat([df_major, df_minor_over], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

   
    df_balanced.to_csv(OUT_PATH, index=False)

    print("Class counts before:", counts.to_dict())
    print("Class counts after:", df_balanced[LABEL].value_counts().to_dict())
    print(f"Saved balanced dataset to: {OUT_PATH}")

if __name__ == "__main__":
    main()
