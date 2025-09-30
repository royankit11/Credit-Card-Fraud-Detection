# Credit Card Fraud Detection - ML Project


## Methods

To prepare the dataset for our model, we need to do some data preprocessing. Since our data contains categorical features, we need to do encoding. We can use one-hot encoding for variables with only a few unique values, such as “transaction category” or “state (location”. For high cardinality variables, like “merchant” or “job”, we can use frequency encoding. This approach avoids the dimensionality explosion that would occur with one-hot encoding. We also need to standardize certain numeric values. Continuous variables such as transaction amount and city population will be standardized so that they contribute equally to model training. This is particularly important for distance-based methods like K-Means clustering and Isolation Forest where consistent feature scales improve convergence. Lastly, we might need to do some minor feature engineering. For instance, extracting day of the week, hour, and weekend/weekday from “transaction date/time” or computing the distance between (lat, long) and (merchant_lat, merchant_long).

Our project will explore both supervised and unsupervised learning methods for fraud detection. For supervised learning, we will use classification models trained on the is_fraud label. Logistic Regression will serve as a baseline due to its interpretability and ability to highlight which features most strongly influence fraud predictions. Random Forest will be applied to capture non-linear feature interactions and provide feature importance measures. We will also evaluate Gradient Boosting, which is well-suited for tabular, imbalanced data and often achieves good results in fraud detection tasks. Together, these models provide a balance between interpretability and predictive power.

In addition to supervised learning, unsupervised methods will be incorporated to detect fraudulent behavior without relying solely on labeled data. K-Means clustering will group transactions into clusters, with outliers flagged as possible fraud cases when they deviate from typical spending behavior. We will also apply Isolation Forest, which isolates outliers in feature space and is particularly effective for rare-event detection. These methods reflect real-world fraud detection challenges, where many fraudulent transactions may remain unlabeled or unseen during training.


