---
layout: page
title: Home
nav-include: true
nav-order: 1
---

## 1. Introduction

Finding online fraud is an essential part of protecting financial institutions and e-commerce sites. The growth of international e-commerce has raised the need for effective models that can detect fraudulent activity. In this study, we aim to explore a machine learning-based approach to predict fraudulent online transactions.

### 1.1 Literature Review

Fraud detection has been widely studied, and several machine learning models have been proposed for detecting fraudulent transactions. One of the most well-known approaches is using supervised learning techniques like Logistic Regression, Decision Trees and Random Forests.
Logistic Regression is valued for simplicity and interpretability [1], while tree-based models handle non-linear interactions and complex feature spaces effectively  [2]. These methods, however, rely heavily on labeled data, which is often limited.

Unsupervised learning methods help when labels are scarce. K-Means and DBSCAN group transactions by similarity, flagging outliers as anomalies [3]. Hybrid strategies that incorporate both supervised and unsupervised techniques have shown promise in recent years. A hybrid strategy could use supervised models to classify the clusters after first training a clustering model to find groups of related transactions [4]. Deep learning techniques, like autoencoders and neural networks, have become popular recently because of their capacity to recognize intricate patterns, but are computationally expensive and require large labeled datasets. [5]

### 1.2 Dataset Description

We use the Kaggle IEEE-CIS fraud detection dataset. It includes two main files: *identity* and *transaction*, linked by *TransactionID*. The transactional data contains features like product codes, card details, and transaction metadata, while the identity data provides device and user information. The target variable *isFraud* indicates whether a transaction is fraudulent.

[Link to Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
