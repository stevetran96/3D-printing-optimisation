import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

# Load data
file = r'C:\Users\steve\Desktop\MECHENG 700\New Data Main - Processed.xlsx'
sheet = 'Combined Data'
features = 15
data = pd.read_excel(file, sheet_name=sheet)
x = data.iloc[:,0:15]  # input: 15 features
y = data.iloc[:,-3]    # output: time in minutes
print(x)
print(y)

# Apply SelectKBest class to score features
bestfeatures = SelectKBest(score_func=mutual_info_regression, k=features)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  # Naming the dataframe columns
print(featureScores.nlargest(features,'Score'))  # Print features and their scores