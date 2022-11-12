import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

title = {'family' : 'normal',
        'size'   : 15}

axis = {'family' : 'normal',
        'size'   : 15}

# Load data
file = r'C:\Users\steve\Desktop\MECHENG 700\New Data Main - Processed.xlsx'
sheet = 'Same Parameters 20%'
features = 1
data = pd.read_excel(file, sheet_name=sheet)
x = data.iloc[:,0:15]  # input: 15 features
y = data.iloc[:,-3]    # output: time in minutes
print(x)
print(y)

# Fitting the model
model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based regressor

# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title('Feature Importance with respect to the response variable Total Minutes', **title)
plt.xlabel('Score of importance', **axis)
plt.ylabel('Feature', **axis)
plt.show()