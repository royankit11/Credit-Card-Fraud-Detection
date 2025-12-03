import pandas as pd

input_path = "augmented_processed.csv"

df = pd.read_csv(input_path)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n_rows = len(df)
n_25 = int(0.25 * n_rows)   # number of rows for 25% file

df_25 = df.iloc[:n_25]      # first 25%
df_75 = df.iloc[n_25:]      # remaining 75%

# ---- 4. Save to two new CSVs ----
df_25.to_csv("test.csv", index=False)
df_75.to_csv("train.csv", index=False)

print("Done: data_25.csv (25%) and data_75.csv (75%) created.")
