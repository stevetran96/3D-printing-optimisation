# FeatureSelection.py: This code implements feature selection algorithms on the Training Data Set to determine
# the importance of each features in terms of their influence over the response variables printing time (minutes)
# and material usage/filament length (metres). Features pairwise correlations are also determined and visualised
# using a heatmap.
#
# Author: Steve Tran
# Date created: 23/9/2019

# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Specifying title and axis label font
title = {'weight' : 'bold',
         'size'   : 15}

axis = {'size'   : 15}


# Load the Training Data Set
file = r'C:\Users\steve\Desktop\Data\Final Data - Processed.xlsx'
sheet = 'Training Full'
# Initialising variables
features = 15
time = -3
length = -1
data = pd.read_excel(file, sheet_name=sheet)
# Separating inputs and outputs
input = data.iloc[:,0:features]       # 15 features
output = data.iloc[:,time]    # printing time (minutes) or material usage (metres)
# Checking if inputs and outputs are correctly imported
print(input)
print(output)

# 1. Univariate Selection: apply SelectKBest class with Mutual Information Regression to score features
bestfeatures = SelectKBest(score_func=mutual_info_regression, k=features)
fit = bestfeatures.fit(input,output)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(input.columns)
# Concatenate two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']        # Naming the dataframe columns
print(featureScores.nlargest(features,'Score'))     # Print features and their scores

# 2. Obtain correlations of each features in data-set
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

# 3. Fitting the model using Extremely Randomised Trees Regression
model = ExtraTreesRegressor()
model.fit(input,output)
print(model.feature_importances_)   # Use inbuilt class feature_importances of tree based regressor
# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=input.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title('Feature Importance with respect to Response Variable', **title)
plt.xlabel('Score of importance', **axis)
plt.ylabel('Feature', **axis)
plt.show()

