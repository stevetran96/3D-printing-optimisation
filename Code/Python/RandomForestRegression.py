# RandomForestRegression.py: This code implements Tree-based algorithms such as Random Forest Regression
# to analyse Training Data Set and build a predictive model to estimate printing time (minutes) and
# material usage/filament length (metres).
# R-Squared and Mean Absolute Percentage Error were used to evaluate the performance of the model.
#
# Author: Steve Tran
# Date created: 25/9/2019

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Using Random Forest Regression to fit the dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
model = RandomForestRegressor()
model.fit(trainInput, trainOutput)

# Testing new shapes
testPredicted = model.predict(testInput)
testR2 = r2_score(testOutput,testPredicted)
testMAPE = mape(testOutput,testPredicted)
print('R-Squared =', testR2)
print('MAPE =', testMAPE)

# Visualising the results
percentageError = 100*((testOutput - testPredicted)/testOutput)
tempplot = np.arange(0,2000,10)
plt.hist(percentageError, 10, facecolor='blue')
plt.title('Random Forest Testing Error')
plt.xlabel('Percentage Error')
plt.ylabel('Number of instances')
plt.grid(True)
plt.show()
plt.plot(testOutput, testPredicted, 'bo', tempplot, tempplot, 'r-')
plt.title('RandomForest: Testing Result - Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()