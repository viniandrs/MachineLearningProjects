from lin_model import LModel as lm
from fold import kfold
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

L = 7
K = 50

def create_model():
    return lm(K, fold)

# Preparing data---------------------------------------------------
df = pd.read_csv('mackeyglass.csv')

# defining the last 750 samples as a test set
test_df = df.loc[len(df) - 750:]

# the rest will be used for training
training_df = df.loc[:len(df) - 750 - 1]

# training_set
training_set = training_df['p'].tolist()

# the fold object
fold = kfold(training_set, 4)
# -----------------------------------------------------------------

t = test_df['t'].tolist()[L + K:]

# the expected values
test_p = test_df['p'].tolist()[L + K:]

# training the model
model = create_model()

# normalizing data
data = test_df['p'].tolist()
norm_data = (data - fold.mean) / fold.stddev

# applying the model
y_hat = model.apply_model_norm(norm_data)
model_p = (y_hat * fold.stddev) + fold.mean

# calculating the error
e = (test_p - model_p)
e2 = e*e
e2_average = np.average(e2)
RMSE = math.sqrt(e2_average)    

print("RMSE: " + str(RMSE))

# Plotting
plt.plot(t, model_p, label = "Model prediction" )
plt.plot(t, test_p, label = "Test samples" )
plt.legend(loc='best', prop={'size': 10})
plt.title("Comparison between the model and test samples")
plt.xlabel("t")
plt.ylabel("P(t)")
plt.grid(True)
plt.show()
