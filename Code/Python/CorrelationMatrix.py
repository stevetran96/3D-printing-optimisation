import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

title = {'family' : 'normal',
        'size'   : 15}

# Load data
file = r'C:\Users\steve\Desktop\MECHENG 700\New Data Main - Processed.xlsx'
sheet = 'New Data'
features = 1
data = pd.read_excel(file, sheet_name=sheet)
x = data.iloc[:,0:15]  # input: 15 features
y = data.iloc[:,-3]    # output: time in minutes
print(x)
print(y)

# Get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.title('Features Pairwise Correlation and with the response varible Total Minutes', **title)
plt.show()