import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

df1 = pd.read_csv("test_identity.csv")
df2 = pd.read_csv("test_transaction.csv")

id_like = [c for c in df1.columns if c.startswith(("id-", "id_"))]
df1 = df1.rename(columns={c: c.replace("_", "-") for c in id_like})

keep1 = ["TransactionID","id-01","id-02","id-05","id-06","id-11","id-13","id-17","id-19","id-20",
        "id-35","id-36","id-37","id-38","DeviceType"]
df1 = df1.loc[:, [c for c in keep1 if c in df1.columns]].copy()

avg_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8","C9", "C10", "C11", "C12", "C13", "C14"]
ext_cols = ["D1", "D2", "D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15"]


if avg_cols:
    df2["avg_c"] = df2[avg_cols].mean(axis=1, skipna=True)
if ext_cols:
    df2["max_d"] = df2[ext_cols].max(axis=1, skipna=True)
    df2["min_d"] = df2[ext_cols].min(axis=1, skipna=True)


df2 = df2.drop(columns=list(set(avg_cols + ext_cols)), errors="ignore")

keep2= ["TransactionID","isFraud","TransactionDT","TransactionAmt","ProductCD","card1", "card2", "card3", "card4","card5", "card6", "P_emaildomain", "avg_c", "max_d","min_d"]

df2 = df2.loc[:, [c for c in keep2 if c in df2.columns]].copy()


KEY = "TransactionID"

df3 = df2.merge(df1, on=KEY, how="left")


cats = ["id-35","id-36","id-37","id-38", "DeviceType", "ProductCD","card4","card6","P_emaildomain"]
cats = [c for c in cats if c in df3.columns]



features = [c for c in (keep1 + keep2) if c in df3.columns]
obj_cols = df3[features].select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in features if c not in obj_cols]

bad = pd.Series(False, index=df3.index)


for c in num_cols:
    bad |= df3[c].isna()


for c in obj_cols:
    s = df3[c].astype("string")
    bad |= (s.isna() | s.str.strip().eq("") | s.str.lower().eq("missing"))

df3 = df3.loc[~bad].copy()



try:  # sklearn >= 1.2
    ohe = OneHotEncoder(handle_unknown="ignore",
                        drop="if_binary",
                        sparse_output=False,
                        dtype=np.uint8)
except TypeError:  # sklearn < 1.2
    ohe = OneHotEncoder(handle_unknown="ignore",
                        drop="if_binary",
                        sparse=False,
                        dtype=np.uint8)

X_ohe = ohe.fit_transform(df3[cats])
enc_cols = ohe.get_feature_names_out(cats)
enc_df  = pd.DataFrame(X_ohe, columns=enc_cols, index=df3.index)

df_enc = pd.concat([df3.drop(columns=cats), enc_df], axis=1)


df_enc.to_csv("processed_test.csv", index=False)
joblib.dump(ohe, "ohe.pkl")