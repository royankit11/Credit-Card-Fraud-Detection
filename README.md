# Credit Card Fraud Detection - ML Project

## 1. Introduction

Finding online fraud is an essential part of protecting financial institutions and e-commerce sites. The growth of international e-commerce has raised the need for effective models that can detect fraudulent activity. In this study, we aim to explore a machine learning-based approach to predict fraudulent online transactions.

### 1.1 Literature Review

Fraud detection has been widely studied, and several machine learning models have been proposed for detecting fraudulent transactions. One of the most well-known approaches is using supervised learning techniques like Logistic Regression, Decision Trees and Random Forests.
Logistic Regression is valued for simplicity and interpretability [1], while tree-based models handle non-linear interactions and complex feature spaces effectively  [2]. These methods, however, rely heavily on labeled data, which is often limited.

Unsupervised learning methods help when labels are scarce. K-Means and DBSCAN group transactions by similarity, flagging outliers as anomalies [3]. Hybrid strategies that incorporate both supervised and unsupervised techniques have shown promise in recent years. A hybrid strategy could use supervised models to classify the clusters after first training a clustering model to find groups of related transactions [4]. Deep learning techniques, like autoencoders and neural networks, have become popular recently because of their capacity to recognize intricate patterns, but are computationally expensive and require large labeled datasets. [5]

### 1.2 Dataset Description

We use the Kaggle IEEE-CIS fraud detection dataset. It includes two main files: *identity* and *transaction*, linked by *TransactionID*. The transactional data contains features like product codes, card details, and transaction metadata, while the identity data provides device and user information. The target variable *isFraud* indicates whether a transaction is fraudulent.

_Link to Dataset:_ (https://www.kaggle.com/competitions/ieee-fraud-detection/data)

## 2. Problem Definition

### 2.1 Problem

Credit card fraud is a rare but high-impact problem, costing billions annually in financial losses and eroding trust in online payments. Because fraudulent transactions are hidden among millions of legitimate ones, detection is difficult: fraud patterns shift over time, labeled data is scarce, and false alarms can frustrate customers. These challenges create a pressing need for models that can accurately distinguish fraud from normal activity while adapting to evolving attack strategies.

### 2.2 Motivation

Accurate fraud detection reduces financial losses, protects cardholders, and sustains trust in digital payments. Missed fraud leads to chargebacks and risks, while false alarms frustrate customers. Therefore, an effective system must balance detection with user experience, while also ensuring fairness across groups and safeguarding data privacy. To address these challenges, we will develop a model that leverages past transaction data to predict the likelihood of fraud.

## 3. Methods

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

### 3.3 Supervised Learning

We'll evaluate multiple supervised models:
* **Logistic Regression** as a baseline, providing interpretability and coefficients that highlight key risk factors.
* **Random Forest** for capturing non-linear feature interactions and generating feature importance.
* **Gradient Boosting**, well-suited for imbalanced tabular data and expected to deliver strong performance.

Together, these methods balance interpretability with predictive power.

### 3.4 Unsupervised Learning

To complement supervised approaches, we use unsupervised anomaly detection:
* **K-Means clustering** groups transactions, flagging those far from cluster centers.
* **Isolation Forest** isolates anomalies effectively in high-dimensional data.
These methods reflect real-world challenges where fraudulent cases are underreported or unseen in training data.

## 4. Results and Discussion

### 4.1 Quantitive Metrics

For this project, we’ll be using several metrics to evaluate our models. We focus on precision, recall, and F1-score, since precision shows how many flagged cases are true fraud, while recall measures coverage of actual fraud cases. F1 balances the two. Finally, the PR-AUC curve will show how precision and recall change across different thresholds, which helps us understand how flexible and reliable the model is in practice.

### 4.2 Project Goals

The objective is to maximize recall while maintaining high precision. A practical target is recall above 95% with precision around 90–95%. While perfect recall and precision are unlikely, a strong F1-score (>0.92) indicates good balance. PR-AUC above 0.85 would confirm robustness across thresholds. Beyond metrics, the system should remain fair across customer groups, respect privacy constraints, and operate efficiently in real time.

### 4.2 Expected Results

We expect Random Forests and Gradient Boosting to deliver strong results, with recall and precision in the 90–95% range. F1-scores will likely land between 0.92 and 0.94, reflecting strong balance. PR-AUC is expected above 0.85, showing reliability under imbalance. While the exact targets (e.g. 99% recall) may not be reached, the models should still provide a powerful fraud detection tool that significantly reduces fraud while minimizing disruptions.

## 5. References

[1]	S. Pamulaparthyvenkata, M. Vishwanath, N. R. Desani, P. Murugesan, and D. Gottipalli, “Non Linear-Logistic Regression Analysis for AI-Driven Medicare Fraud Detection,” International Conference on Distributed Systems, Computer Networks and Cybersecurity, ICDSCNC 2024, 2024, doi: 10.1109/ICDSCNC62492.2024.10939147.

[2]	Q. S. Mirhashemi, N. Nasiri, and M. R. Keyvanpour, “Evaluation of Supervised Machine Learning Algorithms for Credit Card Fraud Detection: A Comparison,” 2023 9th International Conference on Web Research, ICWR 2023, pp. 247–252, 2023, doi: 10.1109/ICWR57742.2023.10139098.

[3]	S. A. Hosseini, F. Grimaccia, A. Niccolai, M. Lorenzo, and F. Casamatta, “Potential Fraud Detection in Carbon Emission Allowance Markets Using Unsupervised Machine Learning Models,” 2024 10th International Conference on Signal Processing and Intelligent Systems, ICSPIS 2024, pp. 33–37, 2024, doi: 10.1109/ICSPIS65223.2024.10931097.

[4]	S. Xu, Y. Cao, Z. Wang, and Y. Tian, “Fraud Detection in Online Transactions: Toward Hybrid Supervised–Unsupervised Learning Pipelines,” 2025 6th International Conference on Electronic Communication and Artificial Intelligence (ICECAI), pp. 470–474, Jun. 2025, doi: 10.1109/ICECAI66283.2025.11171265.

[5]	S. Banoth and K. Madhavi, “A Novel Deep Learning Framework for Credit Card Fraud Detection,” Proceedings of the 2024 13th International Conference on System Modeling and Advancement in Research Trends, SMART 2024, pp. 191–196, 2024, doi: 10.1109/SMART63812.2024.10882509.
 
## Contribution Table

| Name             | Proposal Contributions             |
|------------------|------------------------------------|
| Ankit Roy        | Methods and creating GitHub        |
| Hadi Malik       | Results and Discussion             |
| Yuha Song        | Problem Definition and Gantt Chart |
| Musaddik Hossain | Introduction and References        |
| Connor Priest    | Video Presentation                 |





