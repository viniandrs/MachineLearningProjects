from lin_model import LModel as lm
from fold import kfold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# creates a list with 4 folds from the training set
def create_folds(training_set):
    folds = []
    for i in range(4):
        folds.append(kfold(training_set, i))
        
    return folds

# cross validation using k-fold with k = 4. return an
# array with the respective errors of the folds
def cross_validation(K):
    fold_errors = []
    for i in range(4):
        # creating the model with the fold's training set
        fold = training_folds[i]
        model = lm(K, fold)

        # calculating the fold's error
        fold_error = lm.RMSE_validation(model, fold)
        fold_errors.append(fold_error)

    return np.average(fold_errors)


# Preparing data---------------------------------------------------
df = pd.read_csv('mackeyglass.csv')

# defining the last 750 samples as a test set
test_df = df.loc[len(df) - 750:]
# the rest will be used for training
training_df = df.loc[:len(df) - 750 - 1]

# lists are easier to work with than data frames
test_set = test_df['p'].tolist()
training_set = training_df['p'].tolist()

# list of folds
training_folds = create_folds(training_set)
# -----------------------------------------------------------------

# y-axis
RMSE_array = []

# x-axis
K_array = []

for K in range(2, 51, 2):
    error = cross_validation(K)

    RMSE_array.append(error)
    K_array.append(K)


RMSE_min = min(RMSE_array)
print("minimum RMSE: " + str(RMSE_min))
print("Best K value: " + str(K_array[RMSE_array.index(RMSE_min)]))

# Plotting
plt.plot(K_array, RMSE_array)
plt.title("RMSE x Number of attributes")
plt.xlabel("Number of attributes")
plt.ylabel("Root mean square error")
plt.grid(True)
plt.show()
