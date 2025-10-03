# Credit Card Fraud Detection - ML Project

## 1. Introduction

Finding online fraud is an essential part of protecting financial institutions and e-commerce sites. The growth of international e-commerce has raised the need for effective models that can detect fraudulent activity. In this study, we aim to explore a machine learning-based approach to predict fraudulent online transactions.

### 1.1 Literature Review

Fraud detection has been widely studied, and several machine learning models have been proposed for detecting fraudulent transactions. One of the most well-known approaches is using supervised learning techniques like Logistic Regression, Decision Trees and Random Forests.
Logistic Regression is valued for simplicity and interpretability [1], while tree-based models handle non-linear interactions and complex feature spaces effectively  [2]. These methods, however, rely heavily on labeled data, which is often limited.

Unsupervised learning methods help when labels are scarce. K-Means and DBSCAN group transactions by similarity, flagging outliers as anomalies [3].

Hybrid strategies that incorporate both supervised and unsupervised techniques have shown promise in recent years. A hybrid strategy could use supervised models to classify the clusters after first training a clustering model to find groups of related transactions [4].

Deep learning techniques, like autoencoders and neural networks, have become popular recently because of their capacity to recognize intricate patterns, but are computationally expensive and require large labeled datasets. [5]

### 1.2 Dataset Description

We use the Kaggle IEEE-CIS fraud detection dataset. It includes two main files: *identity* and *transaction*, linked by *TransactionID*. The transactional data contains features like product codes, card details, and transaction metadata, while the identity data provides device and user information. The target variable *isFraud* indicates whether a transaction is fraudulent.

_Link to Dataset:_ (https://www.kaggle.com/competitions/ieee-fraud-detection/data)

## 2. Problem Definition

### 2.1 Problem

This is a binary classification task: estimate the probability of fraud for each transaction and convert it into a yes/no decision by applying a threshold. Inputs are categorical and numeric features from the merged tables. Class imbalance is severe, with fraud comprising a small fraction of records. Validation uses a time-based split to reflect real-world drift in fraud tactics and avoid data leakage. Feature engineering is minimal but includes time-based features (hour, day), rolling counts of recent activity, and imputation for missing values. Continuous features are standardized for distance-based models, though tree models can handle raw values. Because not all fraud is labeled, we also incorporate unsupervised anomaly scores (e.g., Isolation Forest, clustering) to complement supervised probabilities.

### 2.2 Motivation

Early and accurate detection lowers chargebacks, protects cardholders, and avoids false declines that frustrate merchants and customers. Decisions must balance recall (catching fraud) with precision (avoiding bad alerts) under a fixed review capacity (alert budget) for human analysts. For this reason, the project focuses on calibrated probabilities and threshold tuning to meet the review capacity rather than raw accuracy. With heavy imbalance, precision and recall (PR) metrics are the best fit: the area under the PR curve (AUPRC) is the primary score, with recall at fixed precision and precision at fixed recall as operational targets. Unsupervised signals help surface novel fraud tactics as attackers change tactics or as labels are delayed. Fairness and privacy also matter: track precision and decline rates across segments (for example, device types or regions), keep humans in the loop near the decision boundary, and follow data-handling rules. The goal is simple: reduce fraud losses without blocking legitimate customers.

## 3. Methods

### 3.1 Data Preprocessing

Preprocessing includes encoding, scaling, and light feature engineering.
* **Encoding**: One-hot encoding for low-cardinality features (e.g., `ProductCD`, `DeviceType`) and frequency encoding for high-cardinality features (e.g., `card1`, `P_emaildomain`) to avoid dimensionality issues.
* **Scaling**: Standardization for continuous variables like `TransactionAmt`, important for clustering and Isolation Forest.
* **Feature Engineering**: Extracting day, hour, and weekend flags from timestamps; generating aggregate statistics such as transaction counts per card over time windows.

### 3.2 Supervised Learning

We evaluate multiple supervised models:
* **Logistic Regression** as a baseline, providing interpretability and coefficients that highlight key risk factors.
* **Random Forest** for capturing non-linear feature interactions and generating feature importance.
* **Gradient Boosting**, well-suited for imbalanced tabular data and expected to deliver strong performance.

Together, these methods balance interpretability with predictive power.

### 3.3 Unsupervised Learning

To complement supervised approaches, we use unsupervised anomaly detection:
* **K-Means clustering** groups transactions, flagging those far from cluster centers.
* **Isolation Forest** isolates anomalies effectively in high-dimensional data.
These methods reflect real-world challenges where fraudulent cases are underreported or unseen in training data.

## 4. Results and Discussion

### 4.1 Quantitive Metrics

For this project, we’ll be using several metrics to evaluate our models. Precision will tell us the percentage of transactions flagged as fraud that are truly fraudulent, helping us avoid too many false positives. Recall will measure the percentage of actual fraud cases we manage to catch, since missing fraud is especially costly. Because there’s always a tradeoff between precision and recall, we’ll also use the F1-score balance the two. Finally, the PR-AUC curve will give us a broader picture by showing how precision and recall change across different thresholds, which helps us understand how flexible and reliable the model is in practice.

### 4.2 Project Goals

Our main goal is to catch as much fraud as possible, aiming for a recall of around 99%, while also keeping precision high at about 95% so that flagged transactions are almost always real fraud. Hitting both targets at once is tough, but the F1-score helps us measure how close we get to that balance, and we’re aiming for a score above 0.95. For PR-AUC, the goal is to have consistently strong values across thresholds, which would show that the model remains effective even when the data is highly imbalanced. Beyond performance numbers, we also want to make sure the model is fair—avoiding bias against certain groups or regions—and efficient enough to run in real time without unnecessary costs or energy use.

### 4.2 Expected Results

In terms of expected results, we think models like Random Forests and Gradient Boosting will likely deliver recall in the 90–95% range and precision around 90–95% as well. That makes reaching the exact 99% recall and 95% precision targets difficult, but we can still expect strong performance. In practice, the F1-score will probably land between 0.92 and 0.94, which shows a healthy balance between catching fraud and keeping false alarms low. For PR-AUC, we expect results above 0.85, which would demonstrate the model is consistently reliable at different thresholds. Even if the exact goals aren’t fully met, these results would still represent a powerful fraud detection system that catches most fraud while keeping customer disruptions to a minimum, all while staying fair and efficient.

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





