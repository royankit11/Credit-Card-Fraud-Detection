---
layout: page
title: Results
nav-include: true
nav-order: 4
---

### 4.1 Quantitive Metrics

For this project, we’ll be using several metrics to evaluate our models. We focus on precision, recall, and F1-score, since precision shows how many flagged cases are true fraud, while recall measures coverage of actual fraud cases. F1 balances the two. Finally, the ROC-AUC curve will show how the true positive rate varies with the false positive rate across different thresholds, helping us assess the model’s overall discriminative power and robustness.

### 4.2 Project Goals

The objective is to maximize recall while maintaining strong overall classification performance. A practical target is recall above 95% with precision around 90–95%. While perfect recall and precision are unlikely, a strong F1-score (>0.92) indicates good balance. An ROC-AUC in the 0.8–0.9 range would confirm solid discriminative ability across thresholds. Beyond metrics, the system should remain fair across customer groups, respect privacy constraints, and operate efficiently in real time.

### 4.3 Expected Results

We expect Random Forests and Gradient Boosting to deliver strong results, with recall and precision in the 90–95% range. F1-scores will likely land between 0.92 and 0.94, reflecting strong balance. ROC-AUC is expected in the 0.8–0.9 range, indicating robust discriminative ability across thresholds. While the exact targets (e.g., 99% recall) may not be reached, the models should still provide a powerful fraud detection tool that significantly reduces fraud while minimizing disruptions.

### 4.4 Logistic Regression Results and Analysis

For our supervised model, we created a logistic regression model, outputting the probability a given transaction is fraudulent. We created two major models, one using the features from the original dataset, and one with the features from our PCA algorithm. Their results are below:

#### **Regression on Original Features**

**[RESULTS] Logistic Regression (Lightweight)**  

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:-------:|:---------:|:--------:|
| **0** | 0.9886 | 0.8316 | 0.9033 | 113,975 |
| **1** | 0.1366 | 0.7346 | 0.2304 | 4,133 |
| **Accuracy** |  |  | **0.8283** | 118,108 |
| **Macro Avg** | 0.5626 | 0.7831 | 0.5669 | 118,108 |
| **Weighted Avg** | 0.9587 | 0.8283 | 0.8798 | 118,108 |

**ROC AUC:** `0.8639`  

#### **Regression on PCA Features**

**[RESULTS] Logistic Regression on PCA Features**  

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:-------:|:---------:|:--------:|
| **0** | 0.9855 | 0.7598 | 0.8581 | 113,975 |
| **1** | 0.0947 | 0.6925 | 0.1666 | 4,133 |
| **Accuracy** |  |  | **0.7575** | 118,108 |
| **Macro Avg** | 0.5401 | 0.7262 | 0.5123 | 118,108 |
| **Weighted Avg** | 0.9544 | 0.7575 | 0.8339 | 118,108 |

**ROC AUC:** `0.8027`  

From the above statements we are able to learn a lot about how our 2 models performed. For starters, the ROC AUC is a summary metric, telling us how well our model does in general, amongst the entire data set. From this, we see our model that uses the PCA features generates more error than the model that is trained on the original features. This tells us we need to look further into our PCA algorithm, or maybe that we should not be performing PCA, and each feature in the dataset is valuable.

<img width="1054" alt="Logistic Regression Confusion Matrices" src="assets/LR_CM.png" />

The above graphs show a confusion matrix of each of our models, showing how the model predicted fraudulent and nonfraudulent transactions. For the majority of data points, the model behaved well. However, out of roughly 118,000 transactions, the model let 1,097 and 1,271 fraudulent transactions through ourmodel that used the original features and our PCA features, respectively. This equates to roughly 0.01% of all of our transactions making it past our model and turn out to be fraudulent. Depending on the real world implications, or company we decided to work with for this problem, we could tweak this number to better serve client needs.  

<img width="462" alt="Logistic Regression Precision Recall Curve" src="assets/LR_PR.png" />

The above graph is a precision recall curve, plotting the recall values (fraction of actual fraudulent transactions being detected) against the precision values (fraction of predicted frauds that are truly fraud). It shows what the precision values are for all of the different recall values. The AP value stands for average precision, which is mathematically calculated as the area under the curve, but conceptually it represents how precise our model is, while we try to catch more and more fraudulent transactions. The lower AP in our PCA curve tells us that when we try to increase the number of fraudulent transactions caught, this will also increase the number of false positives, which in our scenario could be very costly. A transaction labeled as fraud when it is not, could result in many unhappy customers, frustrated that their transactions are not going through.

The graph above is going to be very helpful in helping us continue to check the number of false positives as we continue to increase the number of fraudulent transactions flagged. As we continue to tweak our model, we will continue to review this chart, and hope to improve our AP scores. An AP score of 0.43 on our orignial model is not terrible given the dataset and the small number of fraudulent transactions we have to work with, yet we still want to improve that number to around 0.6 to ensure we increase correct fraudulent labels. 

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

The above chart is a histogram, showing the distribution of fraudulent and non fraudulent transactions in the dataset, as well as their predicted probability of how likley the transaction was fraudulent. Given the above graph we first see the count in true fraudulent transactions arise around a probability of 0.1, and increase around 0.4. For our first logistic regression model, we chose a simple threshold of 0.5 to determine what transactions would be classified as fraudulent or not. In a real world scenario, we would likely lower that threshold as we want less fraud to be getting through our model. Fraudulent transactions result in huge losses to customers, so it would be better for a bank or financial institution to flag more activity as fraud in order to avoid as much fraud as possible. 

#### Analysis

Given the above visualization on our model, our model provides adequate results to generate a baseline result for predicting fraudulent transactions or not. By using our threshold of 0.50 on our logistic regression model, almost a quarter of the fraudulent transactions are getting through our model and labeled as not fraud. This is definelty an area of our model that needs improvement. While the dataset only has around 4,000 fraudulent transactions available to us to train on, we still wish to stop more fraudulent transactions, and plan to work on this throughout the semester. 

#### Next Steps

Given the state of our model, there are a few steps we can take to look to improve our model. For starters, as mentioned previously we selected a simple 50% threshold to classify a transaction as fraud. While for most scenarios this may be the best value to choose, for this scenario we may want to lower the threshold to classify more variables as fraudulent. Most banks and financial institutions want to ensure they stop as much fraud as possible. For our model, this may come at the cost of mislabeling transactions as fraudulent when they are not, but as a whole we would get fewer false negatives, and would save bank customers lots of money and stress from being fraud victims. We will begin expierementing with this number, to try to reduce the total number of fraudulent transactions getting 'past' the model. Aside from the threshold, we can reexamine our PCA analysis, as the overall accuracy of the model decreased by training on PCA. While this is typical in PCA, we will expierement with getting more features from PCA, and seeing how our accuracy changes. Given we are just running the model locally, we can test adding more PCA features, even though it will increase the model's execution power. 

For a more in depth look at the code generating these images, look into the logistic_regression_analysis notebook file. 

### 4.5 Neural Network (Supervised) Results and Analysis

To make a more advanced supervised model, we architected a neural network, called FraudNet. We created two major models, one using the features from the original dataset, and one with the features from our PCA algorithm. Their results are below:

**[RESULTS] FraudNet (Full Model)**  

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:-------:|:---------:|:--------:|
| **0** | 0.9909 | 0.8608 | 0.9213 | 113,975 |
| **1** | 0.1694 | 0.7830 | 0.2785 | 4,133 |
| **Accuracy** |  |  | **0.8580** | 118,108 |
| **Macro Avg** | 0.5802 | 0.8219 | 0.5999 | 118,108 |
| **Weighted Avg** | 0.9622 | 0.8580 | 0.8988 | 118,108 |

**ROC AUC:** `0.9007`  

#### **FraudNet on PCA Features**

**[RESULTS] FraudNet (PCA Model)**  

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:-------:|:---------:|:--------:|
| **0** | 0.9897 | 0.8348 | 0.9056 | 113,975 |
| **1** | 0.1429 | 0.7595 | 0.2405 | 4,133 |
| **Accuracy** |  |  | **0.8321** | 118,108 |
| **Macro Avg** | 0.5663 | 0.7971 | 0.5731 | 118,108 |
| **Weighted Avg** | 0.9600 | 0.8321 | 0.8824 | 118,108 |

**ROC AUC:** `0.8776`  


#### Confusion Matrices

To visualize performance, we plotted confusion matrices for both models.

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

These matrices show that the Full Model misclassifies fewer fraudulent transactions than the PCA model, consistent with the metrics reported above.

#### ROC Curves

Both models’ ROC curves can be plotted on the same figure for comparison:

<img width="500" alt="Neural Network ROC Curve" src="assets/NN_ROC.png" />

The ROC curves confirm that the Full Model consistently achieves higher true positive rates at the same false positive rates compared to the PCA Model.

#### Precision-Recall Curve

<img width="500" alt="Neural Network Precision Recall Curve" src="assets/NN_PR.png" />

The precision-recall curve shows the trade-off between precision and recall across different thresholds. It illustrates how well the model identifies fraudulent transactions while minimizing false positives.


#### Analysis

FraudNet outperforms logistic regression overall, especially in ROC AUC. The Full Model using original features achieves higher recall and F1-score for fraudulent transactions, capturing more fraud while keeping false positives relatively low. The PCA model performs slightly worse, suggesting that dimensionality reduction may discard important information. The confusion matrices and precision-recall curve confirm these differences, showing the Full Model maintains higher precision across most recall values. Given the severe real-world consequences of undetected fraud, these results highlight the importance of carefully considering which features to include. Overall, FraudNet demonstrates the benefit of a non-linear neural network approach over a linear model.


#### Next Steps

To further improve FraudNet performance, we plan to:

1. **Experiment with threshold tuning** – lowering the probability threshold for classifying fraud to reduce false negatives.

2. **Adjust the network architecture** – explore deeper layers or different hidden sizes to capture more complex patterns.

3. **Incorporate additional regularization or dropout** – to improve generalization without losing predictive power.

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
