# Non-linear regressor

This project improves the simple linear regressor to a non-linear regressor - an Extreme Learning Machine (ELM) [Huang, Zhu & Siew, 2006]. for predicting future values of Mackey-Glass series. The files in this project are used as follow:

- nl_model: Class which creates the non-linear model and base methods (error calculation and predictions)
- fold: Class for managing the k-Fold cross-validation technic for hyperparameter tuning.
- nl_cv_lambda: Tuning of the hyperparameters V (number of transformed attributes) and lambda (regularization factor).
- nl_test: Compares the model built with the optimal values of V and lambda and the test samples.

To visualize the final results, just run nl_test.py.

# References

[Huang, Zhu & Siew, 2006] G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, _Extreme learning machine: theory and applications_. Neurocomputing, vol. 70, pp. 489–501, 2006.
