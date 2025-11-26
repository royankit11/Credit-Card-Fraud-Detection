---
layout: page
title: Methods
nav-include: true
nav-order: 3
---

### 3.1 Data Preprocessing

Preprocessing includes encoding, scaling, and light feature engineering.
* **Encoding**: One-hot encoding for low-cardinality features (e.g., `ProductCD`, `DeviceType`) and frequency encoding for high-cardinality features (e.g., `card1`, `P_emaildomain`) to avoid dimensionality issues.
* **Scaling**: Standardization for continuous variables like `TransactionAmt`, important for clustering and Isolation Forest.
* **Feature Engineering**: Extracting day, hour, and weekend flags from timestamps; generating aggregate statistics such as transaction counts per card over time windows.

### 3.2 PCA

PCA was applied to reduce the high-dimensional V1–V339 feature block while retaining most of the variance, and later performed again on the entire dataset for further dimensionality reduction.
The process:
1. **Group by NaN pattern** - Columns with the same missing-value structure were grouped together.
2. **Within-group reduction** - For each group: large groups used PCA, medium groups kept uncorrelated subsets, and small groups were averaged.
3. **Combined reduced groups** - All reduced outputs were merged back into the dataset.
4. **Final PCA** - A global PCA with 50 components captured 93% of total variance, yielding a compact and efficient feature set for modeling.

For our midterm checkpoint, the data preprocessing algorithm we implemented is **PCA**.

### 3.3 Supervised Learning

We'll evaluate multiple supervised models:
* **Logistic Regression** as a baseline, providing interpretability and coefficients that highlight key risk factors.
* **Random Forest** for capturing non-linear feature interactions and generating feature importance.
* **Gradient Boosting**, well-suited for imbalanced tabular data and expected to deliver strong performance.

For our midterm checkpoint, we implemented the **logistic regression** model.

### 3.4 Unsupervised Learning

To complement supervised approaches, we use unsupervised anomaly detection:
* **K-Means clustering** groups transactions, flagging those far from cluster centers.
* **Isolation Forest** isolates anomalies effectively in high-dimensional data.
These methods reflect real-world challenges where fraudulent cases are underreported or unseen in training data.
