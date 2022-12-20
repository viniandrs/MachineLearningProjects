from fold import kfold
import numpy as np
import math

L = 7
K = 10

# returns an identity matrix with a 0 in [0][0]
def semi_id(size):
    id = np.identity(size)
    id[0][0] = 0
    return id

# generates an array of entries X(n) with K values of
# the series before training_set(n - L)
def entry_array(data, n):
    return data[n - L - K: n - L]

# returns a matrix with every possible array of attributes for
# an array of series values
def build_phi_matrix(data): 
    return [entry_array(data, n) for n in range(L + K, len(data))]


class NLModel:
    # constructor which performs the linear regression to
    # obtain the parameters
    def __init__(self, V, lamb, fold):
        # training data
        data = fold.training_norm

        # obtaining the matrix of effective attributes 
        ##############################

        # matrix of V sets of K random generated parameters
        # from -1 to 1
        self.W = np.random.rand(V, K) * 2 - 1

        # matrix of original attributes
        phi = build_phi_matrix(data)
        
        # matrix of effective attributes
        phi_eff = self.build_phi_eff_matrix(phi)

        # linear regression
        ##############################

        # array of expected outputs
        y = data[L + K:]

        # pseudo-inverse term (transpose(phi).phi)
        pseudo_inv_term = np.dot(np.transpose(phi_eff), phi_eff)

        # regularization term (lambda*semi_I)
        size_of_semi_id = len(pseudo_inv_term)
        semi_identity = semi_id(size_of_semi_id)
        regularization_term = lamb * semi_identity

        # inv(pseudo-inverse + lambda*semi_I).transpose(phi)
        ridge_matrix = np.dot(
            np.linalg.pinv(pseudo_inv_term + regularization_term),
            np.transpose(phi_eff))

        self.params = np.dot(ridge_matrix, y)   
    
    ##############################
    # Methods
    ##############################
    
    # returns a matrix with every possible array of effective attributes for
    # an array of series values
    def build_phi_eff_matrix(self, phi): 
        # matrix of projections of X over W
        projec = np.dot(phi, np.transpose(self.W))

        # matrix of effective attributes
        phi_eff = np.tanh(projec)
        return phi_eff
    
    # calculate the RMSE for a validation set from a given fold
    def RMSE_validation(self, fold):
        # input data
        data = fold.validation_norm
        
        # building the array of expected values
        y_unorm = data[L + K:]
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
        phi = build_phi_matrix(data)
        
        # matrix of effective attributes
        phi_eff = self.build_phi_eff_matrix(phi)
        
        # model's outputs values array
        y_hat = np.dot(phi_eff, self.params)
        
        return y_hat

    
