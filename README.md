# Credit Card Fraud Detection - ML Project

## 1. Introduction

Finding online fraud is an essential part of protecting financial institutions and e-commerce sites. The growth of international e-commerce has raised the need for effective models that can detect fraudulent activity promptly. In this study, we aim to explore a machine learning-based approach to predict the probability of fraudulent online transactions.

### 1.1 Literature Review

Fraud detection has been widely studied, and several machine learning models have been proposed for detecting fraudulent transactions. One of the most well-known approaches is using supervised learning techniques like Logistic Regression, Decision Trees and Random Forests.
Logistic Regression, for instance, is widely used due to its simplicity and interpretability, for any kind of fraud detection [1] . Decision Trees and Random forests have proven effective for fraud detection because they can capture non-linear relationships and handle complex features, such as interactions between categorical variables  [2]. Supervised methods work best when there is a significant amount of labeled data available, which is often challenging.
Unsupervised learning methods are particularly useful in fraud detection when labeled data is scarce or unavailable. Clustering techniques such as K-Means and DBSCAN have been used to detect anomalies by grouping similar transactions together and flagging outliers as potential fraud [3].
Hybrid strategies that incorporate both supervised and unsupervised techniques have shown promise in recent years. A hybrid strategy could use supervised models to classify the clusters after first training a clustering model to find groups of related transactions [4].
Deep learning techniques, like autoencoders and neural networks, have become popular recently because of their capacity to recognize intricate patterns, but they are frequently computationally costly and depend on sizable, labeled datasets for optimal performance. [5]

### 1.2 Dataset Description

The dataset used in this study comes from a Kaggle competition focused on detecting fraudulent online transactions. It includes two main files: *identity* and *transaction*, linked by *TransactionID*. The transactional data contains features like product codes, card details, and transaction metadata, while the identity data provides information about the user's device and personal identity. The target variable *isFraud*, which indicates whether a transaction is fraudulent.

_Link to Dataset:_ (https://www.kaggle.com/competitions/ieee-fraud-detection/data)


## Methods

  To prepare the dataset for our model, we need to do some data preprocessing. Since our data contains categorical features, we need to do encoding. We can use one-hot encoding for variables with only a few unique values, such as `ProductCD` or `DeviceType`. For high cardinality variables, like `card1` or `P_emaildomain`, we can use frequency encoding. This approach avoids the dimensionality explosion that would occur with one-hot encoding. We also need to standardize certain numeric values. Continuous variables such as `TransactionAmt` will be standardized so that they contribute equally to model training. This is particularly important for distance-based methods like **K-Means clustering** and **Isolation Forest** where consistent feature scales improve convergence. Lastly, we might need to do some minor feature engineering. For instance, extracting day of the week, hour, and weekend/weekday from `TransactionDT`. Additional engineered features may include aggregated statistics (e.g. number of transactions per card within a given time window) to capture patterns in user behavior.

  Our project will explore both supervised and unsupervised learning methods for fraud detection. For supervised learning, we will use classification models trained on the `isFraud` label. **Logistic Regression** will serve as a baseline due to its interpretability and ability to highlight which features most strongly influence fraud predictions. **Random Forest** will be applied to capture non-linear feature interactions and provide feature importance measures. We will also evaluate **Gradient Boosting**, which is well-suited for tabular, imbalanced data and often achieves good results in fraud detection tasks. Together, these models provide a balance between interpretability and predictive power.

  In addition to supervised learning, unsupervised methods will be incorporated to detect fraudulent behavior without relying solely on labeled data. **K-Means clustering** will group transactions into clusters, with outliers flagged as possible fraud cases when they deviate from typical spending behavior. We will also apply Isolation Forest, which isolates outliers in feature space and is particularly effective for rare-event detection. These methods reflect real-world fraud detection challenges, where many fraudulent transactions may remain unlabeled or unseen during training.



