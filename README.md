Website is [here](https://royankit11.github.io/Credit-Card-Fraud-Detection/)

## Directory and File Explanation

/src/: All code used for preprocessing, implementing, and analyzing models \
/src/preprocess.py: Implements preprocessing with PCA \
/src/logistic_regression_fraud.py: Implements the logistic regression model \
/src/logistic_regression_analysis.ipynb: Evaluates and presents visualizations for logistic regression \
/src/fraud_net.py: Sets up the architecture of the neural network model \
/src/NN.ipynb: Trains the NN model \
/src/NN_analysis.ipynb: Evaluates and presents visualizations for NN \
/src/model_comparison.ipynb: Compares and presents visualizations for all models \
/src/Kmeans/: All code and data for Kmeans implementation \
/src/Kmeans/Augmentation.py: Balances the dataset by oversampling the minority fraud class \
/src/Kmeans/clean_data.py: Preprocesses and one-hot encodes test data \
/src/Kmeans/k-means.py: Implements K-means model \
/src/Kmeans/kmeans_augmented.py: Runs a command-line K-Means clustering pipeline \
/src/models/: Stores all the implemented ML models \
/src/models/logreg_lightweight.pkl: Logistic regression model on the non-PCA dataset \
/src/models/logreg_pca.pkl: Logistic regression model on the PCA dataset \
/src/models/torch_nn_full.pt: NN model on the non_PCA dataset \
/src/models/torch_nn_pca.pt: NN model on the PCA dataset \
/src/DBSCAN/: All code used for DBSCAN implementation \
/src/DBSCAN/DBSCAN.py: Implements DBSCAN model \
/src/DBSCAN/KNN.py: Computes k-distance curves to estimate DBSCAN clustering parameters \
/src/DBSCAN/split.py: Shuffles dataset and splits into test and training 


