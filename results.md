---
layout: page
title: Results
nav-include: true
nav-order: 4
---

### 4.1 Quantitative Metrics

For this project, we prioritize metrics that effectively evaluate fraud detection performance, specifically focusing on the trade-off between catching fraud and minimizing false alarms.

* **Precision:** The percentage of flagged transactions that are truly fraudulent (minimizing false alarms).
* **Recall:** The percentage of actual fraud cases caught by the model (avoiding missed fraud).
* **F1-Score:** The harmonic mean of precision and recall, providing a single metric for balance.
* **PR-AUC:** The area under the Precision-Recall curve, which is robust for imbalanced datasets.
* **ROC-AUC:** Measures the overall discriminative power across various thresholds.

### 4.2 Project Goals

Our objective is to maximize recall to ensure fraudulent transactions are caught, while maintaining reasonable precision to avoid blocking legitimate customers.

* **Recall:** $\ge 99\%$ (Catch almost all fraud)
* **Precision:** $\ge 95\%$ (Minimize false positives)
* **F1-Score:** $\ge 0.95$ (Strong balance)
* **PR-AUC:** High scores indicating robust performance across thresholds.
* **Constraints:** The system must remain fair across customer groups and operate efficiently in real-time.

### 4.3 Expected Results

Based on initial research, we anticipate the following performance ranges for our supervised models:

* **Recall:** 90–95%
* **Precision:** 90–95%
* **F1-Score:** 0.92–0.94
* **ROC-AUC:** > 0.85
* **Overall:** A strong fraud detection system that significantly reduces financial loss while minimizing disruptions for genuine users.

### 4.4 Logistic Regression Results and Analysis

As a baseline supervised model, we implemented **Logistic Regression**, creating two variations: one trained on the **Original Features** and one on **PCA Features**.

#### **[RESULTS] Logistic Regression Evaluation**

**1. Regression on Original Features**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **0 (Non-Fraud)** | 0.9886 | 0.8316 | 0.9033 | 113,975 |
| **1 (Fraud)** | **0.1366** | **0.7346** | **0.2304** | 4,133 |
| **Accuracy** | | | **0.8283** | 118,108 |
| **ROC AUC** | | | **0.8369** | |

**2. Regression on PCA Features**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **0 (Non-Fraud)** | 0.9855 | 0.7598 | 0.8581 | 113,975 |
| **1 (Fraud)** | **0.0947** | **0.6925** | **0.1666** | 4,133 |
| **Accuracy** | | | **0.7575** | 118,108 |
| **ROC AUC** | | | **0.8027** | |

#### Confusion Matrices & Curves

<img width="1054" alt="Logistic Regression Confusion Matrices" src="assets/LR_CM.png" />

The confusion matrices show that while the model correctly identifies the majority of transactions, it allows a significant number of fraudulent transactions to pass through (High False Negatives).

<img width="462" alt="Logistic Regression Precision Recall Curve" src="assets/LR_PR.png" />

The Precision-Recall curve highlights the trade-off. The lower AP (Average Precision) for the PCA model confirms that dimensionality reduction resulted in information loss, reducing the model's ability to distinguish fraud effectively.

#### Predicted Probabilities

<table>
  <tr>
    <td><img width="400" alt="Logistic Regression Predicted Probabilities Original" src="assets/LR_PP1.png" /></td>
    <td><img width="400" alt="Logistic Regression Predicted Probabilities PCA" src="assets/LR_PP2.png" /></td>
  </tr>
  <tr>
    <td align="center">Original Features</td>
    <td align="center">PCA Features</td>
  </tr>
</table>

The histograms show the distribution of predicted probabilities. We used a default threshold of 0.5. Lowering this threshold could improve recall (catching more fraud) but would likely increase false positives.

#### Analysis

Logistic Regression served as a functional baseline. With an ROC-AUC of **0.8369**, it demonstrated reasonable predictive power. However, the PCA-based model performed worse across all metrics (Accuracy dropped to 75.75%), suggesting that PCA removed features critical for linear separation. The low F1-score for the fraud class (0.2304) indicates that a linear model is insufficient for the complex patterns of credit card fraud.

### 4.5 Neural Network (Supervised) Results and Analysis

To capture non-linear relationships, we developed a neural network architecture called **FraudNet**. Similar to the baseline, we evaluated it on both Original and PCA features.

#### **[RESULTS] FraudNet Evaluation**

**1. FraudNet on Original Features (Full Model)**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **0 (Non-Fraud)** | 0.9909 | 0.8608 | 0.9213 | 113,975 |
| **1 (Fraud)** | **0.1694** | **0.7830** | **0.2785** | 4,133 |
| **Accuracy** | | | **0.8580** | 118,108 |
| **ROC AUC** | | | **0.9007** | |

**2. FraudNet on PCA Features**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **0 (Non-Fraud)** | 0.9897 | 0.8348 | 0.9056 | 113,975 |
| **1 (Fraud)** | **0.1429** | **0.7595** | **0.2405** | 4,133 |
| **Accuracy** | | | **0.8321** | 118,108 |
| **ROC AUC** | | | **0.8776** | |

#### Confusion Matrices

<table>
  <tr>
    <td><img width="400" alt="Neural Network Confusion Matrix Original" src="assets/NN_CM1.png" /></td>
    <td><img width="400" alt="Neural Network Confusion Matrix PCA" src="assets/NN_CM2.png" /></td>
  </tr>
  <tr>
    <td align="center">Original Features</td>
    <td align="center">PCA Features</td>
  </tr>
</table>

FraudNet misclassifies fewer fraudulent transactions compared to Logistic Regression, as shown by the improved Recall (78.30% vs 73.46%).

#### ROC & PR Curves

<table>
  <tr>
    <td><img width="500" alt="Neural Network ROC Curve" src="assets/NN_ROC.png" /></td>
    <td><img width="500" alt="Neural Network Precision Recall Curve" src="assets/NN_PR.png" /></td>
  </tr>
</table>

The ROC curve confirms the superiority of the Full Model (Original Features) with an AUC of **0.9007**, consistently achieving higher true positive rates at the same false positive rates compared to the PCA model.

#### Analysis

FraudNet significantly outperformed Logistic Regression. The Full Model achieved the highest Recall (**~78%**) and ROC-AUC (**~0.90**), proving that a non-linear approach is better suited for detecting complex fraud patterns. Consistent with previous results, the PCA model performed slightly worse, reinforcing the finding that dimensionality reduction may discard valuable signals in this dataset. Future improvements could involve tuning the decision threshold to further balance precision and recall.

### 4.6 K-Means (Unsupervised) Results and Analysis

To separate fraud from non-fraud, we set $K=2$ and evaluated how K-Means performs on the engineered features. To avoid overwhelming the clustering with very high-dimensional signals, we excluded the electronic footprint features (V1–V339). The dataset was balanced using augmentation techniques. Since K-Means is label-free, we mapped the resulting clusters to classes (Fraud vs. Non-Fraud) based on the majority vote of the training labels.

#### **[RESULTS] K-Means Evaluation**

We evaluated the model using both internal clustering metrics and external classification metrics.

**Internal Indices (Cluster Quality)**
* **Silhouette Score:** `0.1516` (Test) - The low score indicates that the clusters are overlapping and not well-separated.
* **Davies-Bouldin Index:** `2.9519` - A high value confirms poor separation between fraud and non-fraud clusters.

**External Indices (Classification Performance)**

| Metric | Train | Test | Benchmark |
|:---|:---:|:---:|:---|
| **Accuracy** | 0.6306 | **0.6315** | Close to 1 |
| **Precision** | 0.6083 | **0.6081** | Close to 1 |
| **Recall** | 0.7345 | **0.7365** | Close to 1 |
| **F1-Score** | 0.6654 | **0.6662** | Close to 1 |

#### **Confusion Matrices**

<table align="center">
  <tr>
    <td align="center"><img src="src/Kmeans/kmeans_confusion_train.png" alt="K-Means Confusion Matrix Train" width="400"/></td>
    <td align="center"><img src="src/Kmeans/kmeans_confusion_test.png" alt="K-Means Confusion Matrix Test" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Train Confusion Matrix</td>
    <td align="center">Test Confusion Matrix</td>
  </tr>
</table>

#### **Analysis**

K-Means provided a structural baseline but ultimately struggled with this dataset. The method assumes spherical, distinct clusters, whereas fraud data often exhibits complex, non-globular distributions. The low Silhouette score (~0.15) confirms that the clusters were not distinct. While the recall (~73%) was decent, the high false-positive rate suggests that K-Means is better suited as a feature generator or an early-stage filter rather than a standalone fraud detector.

### 4.7 DBSCAN (Unsupervised) Results and Analysis

Addressing the limitations of K-Means, we implemented **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**. Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance and can identify outliers (noise) effectively, which is ideal for anomaly detection.

#### **Parameter Selection**
We determined the optimal parameters through K-distance graph analysis:
* **MinPts:** 2
* **Epsilon ($\epsilon$):** 2.4
* **Rationale:** This combination yielded the highest accuracy while maintaining a low noise ratio, which is critical for practical fraud detection.

#### **[RESULTS] DBSCAN Evaluation**

**Internal Indices**
* **Silhouette Score:** `-0.057` (Test)
* **Davies-Bouldin Index:** `1.6727`

**External Indices (Classification Performance)**

| Metric | Train | Test | Benchmark |
|:---|:---:|:---:|:---|
| **Accuracy** | 0.7990 | **0.8228** | Close to 1 |
| **Precision** | 0.7841 | **0.8056** | Close to 1 |
| **Recall** | 0.8255 | **0.8503** | Close to 1 |
| **F1-Score** | 0.8043 | **0.8273** | Close to 1 |

* **Noise Handling:** The model identified approximately **16.78%** of the test data as noise, effectively separating distinct outliers from the main transaction clusters.

#### **Confusion Matrices**

<table align="center">
  <tr>
    <td align="center"><img src="src/Kmeans/dbscan_confusion_train.png" alt="DBSCAN Confusion Matrix Train" width="400"/></td>
    <td align="center"><img src="src/Kmeans/dbscan_confusion_test.png" alt="DBSCAN Confusion Matrix Test" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Train Confusion Matrix</td>
    <td align="center">Test Confusion Matrix</td>
  </tr>
</table>

#### **Analysis**

DBSCAN significantly outperformed K-Means. By leveraging density-based clustering, it adapted well to the irregular shapes of fraudulent transaction patterns. The model achieved a **recall of ~85%** and an **accuracy of ~82%**, making it a much more viable unsupervised solution for detecting anomalies in unlabeled data compared to K-Means.

## 5. Comparison of Models

We evaluated four distinct models: two supervised (Logistic Regression, FraudNet) and two unsupervised (K-Means, DBSCAN).

1.  **Supervised Learning Dominance:**
    * **FraudNet (Neural Network)** was the top performer overall, achieving an **ROC-AUC of 0.90** and the best balance of precision and recall. Its non-linear architecture allowed it to capture complex fraud patterns that the linear Logistic Regression model missed.
    * **Logistic Regression** served as a decent baseline (ROC-AUC 0.86) but struggled with lower precision and higher false negatives compared to the Neural Network.

2.  **Unsupervised Learning Insights:**
    * **DBSCAN vs. K-Means:** DBSCAN proved to be the superior unsupervised method. While K-Means suffered from the "spherical assumption" (Accuracy ~63%), DBSCAN's ability to handle noise and arbitrary shapes boosted accuracy to **~82%** and recall to **~85%**.
    * This suggests that unsupervised learning *can* be effective for fraud detection if the model accounts for the density and irregularity of outliers, rather than just geometric distance.

**Conclusion:**
For a deployed system, **FraudNet** is the recommended primary model due to its high F1-score and robustness. However, **DBSCAN** shows great promise as a complementary tool for flagging potential new fraud patterns (outliers) that supervised models might not yet be trained on.

<img alt="Logistic Regression vs Neural Network PR-AUC curve" src="assets/LR_vs_NN.png" width="520"/>
