from nl_model import NLModel as nlm
from fold import kfold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# creates a list with 4 folds from the training set
def create_folds(training_set):
    folds = []
    for i in range(4):
        folds.append(kfold(training_set, i))
        
    return folds

# cross validation using k-fold with k = 4. return an
# array with the respective errors of the folds
def cross_validation(V, lamb):
    fold_errors = []
    for i in range(4):
        # creating the model with the fold's training set
        fold = training_folds[i]
        model = nlm(V, lamb, fold)

        # calculating the fold's error
        fold_error = nlm.RMSE_validation(model, fold)
        fold_errors.append(fold_error)

    return np.average(fold_errors)


def find_best_lambda(V):
    errors = []
    lambdas = []
    for lamb_int in range(10, 30, 5):
        lamb = math.pow(10, -lamb_int)

        error = cross_validation(V, lamb)

        errors.append(error)
        lambdas.append(lamb)

    index = errors.index(min(errors))

    return lambdas[index], errors[index]


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
V_array = []

# (V, best_lambda)
V_and_lambda = []

for V in range(40, 1001, 40):
    lamb, V_error = find_best_lambda(V)
    V_and_lambda.append((V, lamb))

    RMSE_array.append(V_error)
    V_array.append(V)
    
print(V_and_lambda)

RMSE_min = min(RMSE_array)
print("minimum RMSE: " + str(RMSE_min))
print("Best V: " + str(V_array[RMSE_array.index(RMSE_min)]))
print("Best lambda: " + str(V_and_lambda[RMSE_array.index(RMSE_min)][1]))

# Plotting
plt.plot(V_array, RMSE_array)
plt.title("RMSE x Número de atributos efetivos")
plt.xlabel("No. de atributos efetivos")
plt.ylabel("Sqrt do erro quad. medio")
plt.grid(True)
plt.show()
