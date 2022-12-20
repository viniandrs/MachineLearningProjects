# Non-linear classifier - k nearest neighbors

Non-linear classifier applied to the BreastMNIST dataset. Image labels are either 0 (cancer) or 1 (healthy). It was used the SciKit Learn library just like in the linear classifier. The files in this project are used as follow:

- knn_cv: k-fold cross validation for hyperparameter tuning. 
- knn_test: Compares the model built with the optimal values of the number of neighbors and the Minkowski order and the test samples.

To visualize the final results, just run knn_test.py.
