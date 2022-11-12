import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Load data
file = r'C:\Users\steve\Desktop\MECHENG 700\New Data Main - Processed.xlsx'
sheet = 'Combined Data'
features = 1
data = pd.read_excel(file, sheet_name=sheet)
x = data.iloc[:,1:16]  # input: 15 features
y = data.iloc[:,-3]    # output: time in minutes
print(x)
print(y)
# Feature Selection
model = LinearRegression()
rfe = RFE(model, features)
fitted = rfe.fit(x, y)
print("Num Features: %d" % fitted.n_features_)
print("Selected Features: %s" % fitted.support_)
print("Feature Ranking: %s" % fitted.ranking_)