# PolynomialRegression.py: This code implements polynomial regression to analyse Training Data Set and build a
# predictive model to estimate printing time (minutes) and material usage/filament length (metres).
# R-Squared and Mean Absolute Percentage Error were used to evaluate the performance of the model.
#
# Author: Steve Tran
# Date created: 24/9/2019

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define MAPE as a performance metrics
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the Training Data Set
file = r'C:\Users\steve\Desktop\Data\Final Data - Processed.xlsx'
trainSheet = 'Training Full'
testSheet = 'Testing Full'
# Initialising variables
features = 15
time = -3
length = -1
trainData = pd.read_excel(file, sheet_name=trainSheet)
testData = pd.read_excel(file, sheet_name=testSheet)
# Separating inputs and outputs
trainInput = trainData.iloc[:,0:15]       # 15 features
trainOutput = trainData.iloc[:,time]    # printing time (minutes) or material usage (metres)
testInput = testData.iloc[:,0:15]       # 15 features
testOutput = testData.iloc[:,time]    # printing time (minutes) or material usage (metres)


# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
# Calculating polynomial terms
poly = PolynomialFeatures(degree = 3)
trainPoly = poly.fit_transform(trainInput)
poly.fit(trainPoly, trainOutput)
model = LinearRegression()
model.fit(trainPoly, trainOutput)

# Testing new shapes
testPoly = poly.fit_transform(testInput)
testPredicted = model.predict(testPoly)
testR2 = r2_score(testOutput,testPredicted)
testMAPE = mape(testOutput,testPredicted)
print('R-Squared =', testR2)
print('MAPE =', testMAPE)

# Visualising the results
percentageError = 100*((testOutput - testPredicted)/testOutput)
tempplot = np.arange(0,2000,10)
plt.hist(percentageError, 10, facecolor='blue')
plt.title('Polynomial Regression: Histogram of Testing Prediction Error')
plt.xlabel('Percentage Error')
plt.ylabel('Number of instances')
plt.grid(True)
plt.show()
plt.plot(testOutput, testPredicted, 'bo', tempplot, tempplot, 'r-')
plt.title('Polynomial Regression: Testing Result - Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

