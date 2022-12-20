import numpy as np

NFOLDS = 4

def normalize(data):
    if len(data) <= 1:
        return data

    stddev = np.std(data) + 0.000000001
    mean = np.average(data)
    data_norm = (data - mean)/stddev
    
    return data_norm, mean, stddev
    

class kfold:
    def __init__(self, data, fold_index):                       
        # RAW DATA
        step = int(len(data)/(2*NFOLDS))
        data_size = 2*NFOLDS*step
        
        validation_start = int(data_size/2 + fold_index*step + 1)
        validation_end = int(validation_start + step + 1)
        
        # test
        if fold_index >= NFOLDS:
            ## UNORMALIZED DATA
            self.training_data = data[:validation_end]

            # NORMALIZED DATA - to be used in the training 
            self.training_norm, self.mean, self.stddev = normalize(self.training_data)
        
        # cross validation    
        else:
            # UNORMALIZED DATA
            self.training_data = data[:validation_start]
            self.validation_data = data[validation_start:validation_end]

            # NORMALIZED DATA - to be used in the training 
            self.training_norm, self.mean, self.stddev = normalize(self.training_data)
            self.validation_norm = (self.validation_data - self.mean) / self.stddev
        
        
        
            
