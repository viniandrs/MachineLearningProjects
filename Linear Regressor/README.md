# Linear regressor

A simple linear regressor for predicting future values of Mackey-Glass series. The files in this project are used as follow:

- lin_model: Class which creates the linear model and base methods (error calculation and predictions)
- fold: Class for managing the k-Fold cross-validation technic for hyperparameter tuning.
- lin_cv_k: Tuning of the hyperparameter K (number of series' past values used as attributes).
- lin_test: Compares the model built with the optimal value of K and the test samples.

To visualize the final results, just run lin_test.py.