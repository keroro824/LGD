import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset

train_x = "/Users/beidichen/Documents/2018sp/Ironman_LSD/test/slice_x.learn"  
train_y = "/Users/beidichen/Documents/2018sp/Ironman_LSD/test/slice_y.learn"
test_x = "/Users/beidichen/Documents/2018sp/Ironman_LSD/test/slice_x.test"
test_y = "/Users/beidichen/Documents/2018sp/Ironman_LSD/test/slice_y.test"

Xtrain = np.genfromtxt(train_x, dtype=float, delimiter=" ")
Ytrain = np.genfromtxt(train_y, dtype=float, delimiter=" ")
Xtest = np.genfromtxt(test_x, dtype=float, delimiter=" ")
Ytest = np.genfromtxt(test_y, dtype=float, delimiter=" ")


# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Train the model using the training sets
regr.fit(Xtrain, Ytrain)

# Make predictions using the testing set
y_pred = regr.predict(Xtest)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Ytest, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Ytest, y_pred))
