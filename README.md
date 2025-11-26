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

## 4. Results and Discussion

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

<img width="1054" alt="Screenshot 2025-11-06 at 8 58 13 AM" src="https://github.gatech.edu/user-attachments/assets/9f65b333-ef2a-45bb-8a06-e0e24ce6844d" />

The above graphs show a confusion matrix of each of our models, showing how the model predicted fraudulent and nonfraudulent transactions. For the majority of data points, the model behaved well. However, out of roughly 118,000 transactions, the model let 1,097 and 1,271 fraudulent transactions through ourmodel that used the original features and our PCA features, respectively. This equates to roughly 0.01% of all of our transactions making it past our model and turn out to be fraudulent. Depending on the real world implications, or company we decided to work with for this problem, we could tweak this number to better serve client needs.  

<img width="462" alt="Screenshot 2025-11-06 at 9 34 53 AM" src="https://github.gatech.edu/user-attachments/assets/5ed426ce-6133-4459-a120-3bfee802f975" />

The above graph is a precision recall curve, plotting the recall values (fraction of actual fraudulent transactions being detected) against the precision values (fraction of predicted frauds that are truly fraud). It shows what the precision values are for all of the different recall values. The AP value stands for average precision, which is mathematically calculated as the area under the curve, but conceptually it represents how precise our model is, while we try to catch more and more fraudulent transactions. The lower AP in our PCA curve tells us that when we try to increase the number of fraudulent transactions caught, this will also increase the number of false positives, which in our scenario could be very costly. A transaction labeled as fraud when it is not, could result in many unhappy customers, frustrated that their transactions are not going through.

The graph above is going to be very helpful in helping us continue to check the number of false positives as we continue to increase the number of fraudulent transactions flagged. As we continue to tweak our model, we will continue to review this chart, and hope to improve our AP scores. An AP score of 0.43 on our orignial model is not terrible given the dataset and the small number of fraudulent transactions we have to work with, yet we still want to improve that number to around 0.6 to ensure we increase correct fraudulent labels. 

<img width="628" alt="Screenshot 2025-11-06 at 9 35 15 AM" src="https://github.gatech.edu/user-attachments/assets/9213e0d2-266f-4cea-9691-48efcca0cfc1" />

<img width="628" alt="Screenshot 2025-11-06 at 9 35 27 AM" src="https://github.gatech.edu/user-attachments/assets/8426dd0d-78de-4b30-b1d5-248619c778b6" />

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

**ROC AUC:** `0.9007`  


#### Confusion Matrices

To visualize performance, we plotted confusion matrices for both models.

<img width="350" alt="Screenshot 2025-11-26 at 11 51 05 AM" src="https://github.gatech.edu/user-attachments/assets/e826c31b-3e14-4b58-aaaa-4281b91ec87f" />
<img width="350" alt="Screenshot 2025-11-26 at 11 51 12 AM" src="https://github.gatech.edu/user-attachments/assets/729aba49-a786-4f6e-8195-f20755209bbe" />

These matrices show that the Full Model misclassifies fewer fraudulent transactions than the PCA model, consistent with the metrics reported above.

#### ROC Curves

Both models’ ROC curves can be plotted on the same figure for comparison:

<img width="691" alt="Screenshot 2025-11-26 at 11 50 43 AM" src="https://github.gatech.edu/user-attachments/assets/8a1f190d-6ea1-4ae1-93ff-38a9003937a5" />

The ROC curves confirm that the Full Model consistently achieves higher true positive rates at the same false positive rates compared to the PCA Model.

#### Precision-Recall Curve

<img width="689" alt="Screenshot 2025-11-26 at 11 50 56 AM" src="https://github.gatech.edu/user-attachments/assets/ed7c1285-a3de-443d-8176-0096a97c14a6" />

The precision-recall curve shows the trade-off between precision and recall across different thresholds. It illustrates how well the model identifies fraudulent transactions while minimizing false positives.


#### Analysis

FraudNet outperforms logistic regression overall, especially in ROC AUC. The Full Model using original features achieves higher recall and F1-score for fraudulent transactions, capturing more fraud while keeping false positives relatively low. The PCA model performs slightly worse, suggesting that dimensionality reduction may discard important information. The confusion matrices and precision-recall curve confirm these differences, showing the Full Model maintains higher precision across most recall values. Given the severe real-world consequences of undetected fraud, these results highlight the importance of carefully considering which features to include. Overall, FraudNet demonstrates the benefit of a non-linear neural network approach over a linear model.


#### **Next Steps**

To further improve FraudNet performance, we plan to:

1. **Experiment with threshold tuning** – lowering the probability threshold for classifying fraud to reduce false negatives.

2. **Adjust the network architecture** – explore deeper layers or different hidden sizes to capture more complex patterns.

3. **Incorporate additional regularization or dropout** – to improve generalization without losing predictive power.


### 4.6 K-Means (Unsupervised) Results and Analysis

To separate fraud from non fraud we set k=2 and evaluate how K Means performs on the engineered features in Section 3.1. To avoid overwhelming the clustering with very high dimensional signals, we exclude the electronic footprint features V1–V339. After feature engineering the data were imbalanced at about 10:1 non fraud to fraud, so we applied data augmentation to balance the train and test splits. Because K Means is label free, we fit on the train split, map clusters to classes (isFraud ∈ {0,1}) by majority vote on the training labels, and apply that mapping to both splits. The findings for training and testing are reported below.

---

#### [RESULTS] K Means ⇒ Majority Label (Train)

| Class | Precision | Recall | F1 Score | Support |
|:--|:--:|:--:|:--:|--:|
| 0 (Non fraud) | 0.6657 | 0.5259 | 0.5876 | 74,976 |
| 1 (Fraud)     | 0.6082 | 0.7359 | 0.6660 | 74,976 |
| Accuracy |  |  | 0.6309 | 149,952 |
| Macro Avg | 0.6369 | 0.6309 | 0.6268 | 149,952 |
| Weighted Avg | 0.6369 | 0.6309 | 0.6268 | 149,952 |

<img alt="Confusion Matrix K Means Train" src="src/Kmeans/kmeans_confusion_train.png" width="520"/>

#### [RESULTS] K Means ⇒ Majority Label (Test)

| Class | Precision | Recall | F1 Score | Support |
|:--|:--:|:--:|:--:|--:|
| 0 (Non fraud) | 0.6643 | 0.5288 | 0.5888 | 24,992 |
| 1 (Fraud)     | 0.6086 | 0.7327 | 0.6649 | 24,992 |
| Accuracy |  |  | 0.6308 | 49,984 |
| Macro Avg | 0.6365 | 0.6308 | 0.6269 | 49,984 |
| Weighted Avg | 0.6365 | 0.6308 | 0.6269 | 49,984 |

<img alt="Confusion Matrix K Means Test" src="src/Kmeans/kmeans_confusion_test.png" width="520"/>

Train and test behave similarly, so the unsupervised pipeline does not appear to overfit. In both splits the fraud class has recall about 0.73 and precision about 0.61. This means many frauds are caught with more false alarms. False negatives are lower. Overall accuracy is about 0.631.

---

#### Precision and recall behavior (score distributions)

We convert distance to centroid into a normalized fraud score in [0,1]. Larger values indicate more anomalous behavior. The histograms show that frauds shift toward higher anomaly scores, but there is still large overlap, which explains moderate precision at high recall.

<img alt="Predicted Scores K Means Train" src="src/Kmeans/kmeans_score_hist_train.png" width="520"/>
<img alt="Predicted Scores K Means Test"  src="src/Kmeans/kmeans_score_hist_test.png"  width="520"/>

---

#### Internal cluster quality measures (label free)

We report Silhouette (↑), Calinski–Harabasz (↑), Davies–Bouldin (↓), and BetaCV (↓) on standardized features for train and test. All metrics are plotted on one linear axis.

<img alt="K Means Internal Metrics" src="src/Kmeans/kmeans_internal_metrics_linear_single.png" width="820"/>

---

#### Analysis

K Means did not work fine in this setting. It is a centroid based method that best separates roughly circular or spherical clusters. Our data likely have a more complex and non spherical distribution. This explains the higher false positives and false negatives seen in the confusion matrices and the internal measures. Silhouette is low, DB and BetaCV are higher, and CH is only modest. These patterns indicate overlapping clusters rather than clean separation. K Means should be used as a baseline or as a feature generator and early filter, not as a standalone fraud detector.

#### Next steps

We will test k greater than 2 with stable k means++ restarts and sanity checks across resamples such as ARI and NMI. We will try dimensionality reduction such as PCA or UMAP before clustering. We will also evaluate Gaussian Mixture Models and DBSCAN or HDBSCAN to capture non spherical structure. Finally, we will feed distance to centroid scores into the supervised model and calibrate thresholds to control the false positive rate while monitoring centroid drift and internal measures over time.


## 5. References

[1]	S. Pamulaparthyvenkata, M. Vishwanath, N. R. Desani, P. Murugesan, and D. Gottipalli, “Non Linear-Logistic Regression Analysis for AI-Driven Medicare Fraud Detection,” International Conference on Distributed Systems, Computer Networks and Cybersecurity, ICDSCNC 2024, 2024, doi: 10.1109/ICDSCNC62492.2024.10939147.

[2]	Q. S. Mirhashemi, N. Nasiri, and M. R. Keyvanpour, “Evaluation of Supervised Machine Learning Algorithms for Credit Card Fraud Detection: A Comparison,” 2023 9th International Conference on Web Research, ICWR 2023, pp. 247–252, 2023, doi: 10.1109/ICWR57742.2023.10139098.

[3]	S. A. Hosseini, F. Grimaccia, A. Niccolai, M. Lorenzo, and F. Casamatta, “Potential Fraud Detection in Carbon Emission Allowance Markets Using Unsupervised Machine Learning Models,” 2024 10th International Conference on Signal Processing and Intelligent Systems, ICSPIS 2024, pp. 33–37, 2024, doi: 10.1109/ICSPIS65223.2024.10931097.

[4]	S. Xu, Y. Cao, Z. Wang, and Y. Tian, “Fraud Detection in Online Transactions: Toward Hybrid Supervised–Unsupervised Learning Pipelines,” 2025 6th International Conference on Electronic Communication and Artificial Intelligence (ICECAI), pp. 470–474, Jun. 2025, doi: 10.1109/ICECAI66283.2025.11171265.

[5]	S. Banoth and K. Madhavi, “A Novel Deep Learning Framework for Credit Card Fraud Detection,” Proceedings of the 2024 13th International Conference on System Modeling and Advancement in Research Trends, SMART 2024, pp. 191–196, 2024, doi: 10.1109/SMART63812.2024.10882509.
 
## Contribution Table

| Name             | Midterm Contributions              |
|------------------|------------------------------------|
| Ankit Roy        | Implementing logistic regression   |
| Hadi Malik       | Preprocessing with PCA             |
| Yuha Song        | Evaluating K Means                 |
| Musaddik Hossain | Implementing K Means               |
| Connor Priest    | Evaluating logistic regression     |


## Directory and File Explanation

/src/: All code used for preprocessing, implementing, and analyzing models \
/src/preprocess.py: Implements preprocessing with PCA \
/src/logistic_regression_fraud.py: Implements the logistic regression model \
/src/logistic_regression_analysis.ipynb: Evaluates and presents visualizations for logistic regression \
/src/Kmeans/: All code and data for Kmeans implementation \
/src/Kmeans/Augmentation.py: Balances the dataset by oversampling the minority fraud class \
/src/Kmeans/clean_data.py: Preprocesses and one-hot encodes test data \
/src/Kmeans/k-means.py: Implements K-means model \
/src/Kmeans/kmeans_augmented.py: Runs a command-line K-Means clustering pipeline \
/src/models/: Stores all the implemented ML models \
/src/models/logreg_lightweight.pkl: Logistic regression model on the non-PCA dataset \
/src/models/logreg_pca.pkl: Logistic regression model on the PCA dataset




