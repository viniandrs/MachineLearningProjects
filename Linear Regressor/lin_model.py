from fold import kfold
import numpy as np
import math

L = 7

class LModel:
    # constructor which performs the linear regression to
    # obtain the parameters
    def __init__(self, n_attributes, fold):
        self.K = n_attributes
        
        # training data
        data = fold.training_norm

        # matrix of original attributes
        phi = self.build_phi_matrix(data)

        # linear regression
        ##############################

        # array of expected outputs
        y = data[L + self.K:]

        # Equation 3
        pseudo_inv = np.dot(
            np.linalg.pinv(np.dot(np.transpose(phi), phi)),
            np.transpose(phi))

        self.params = np.dot(pseudo_inv, y)

    ##############################
    # Methods
    ##############################

    # generates an array of entries X(n) with K values of
    # the series before training_set(n - L)

    def entry_array(self, data, n):
        return data[n - L - self.K: n - L]

    # returns a matrix with every possible array of attributes for
    # an array of series values

    def build_phi_matrix(self, data):
        return [self.entry_array(data, n) for n in 
                range(L + self.K, len(data))]

    # calculate the RMSE for a validation set from a given fold
    def RMSE_validation(self, fold):
        # input data
        data = fold.validation_norm

        # building the array of expected values
        y_unorm = data[L + self.K:]
        y = (y_unorm * fold.stddev) + fold.mean

        # applying the model
        y_hat_unorm = self.apply_model_norm(data)
        y_hat = (y_hat_unorm * fold.stddev) + fold.mean

        # error array
        e = y - y_hat
        e2 = e * e

        # calculating the average square error
        e2_average = np.average(e2)

        # finally the RMSE
        RMSE = math.sqrt(e2_average)

        return RMSE

    # apply the optimized model to a given set of inputs
    # and return a list with the normalized predictions
    def apply_model_norm(self, data):
        # matrix of original attributes
        phi = self.build_phi_matrix(data)

        # model's outputs values array
        y_hat = np.dot(phi, self.params)

        return y_hat
