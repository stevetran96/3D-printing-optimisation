This README file includes the description of the files and folders located in Code\MATLAB
as well as explanation of different versions where applicable.

Versions:  
MATLAB R2019a

This folder contains the latest versions of the MATLAB code to build predictive models using machine learning for my project,
as well as subfolders for old versions.

=================================

FILES:  
SupportEstimation.m - The latest code to provide an estimation of volumn of support structure required for a model at a particular orientation on the print bed.  
Input: spreadsheets located in Model\Cross-sectional Area\File Name - Orientation.xlsx

neuralNetwork_Final.m - The latest code to build predictive model with feedforward-backpropagation neural network to estimate printing time
and material usage for trained and untrained CAD models at new printing parameters.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- predict printing time and material usage for validation of trained CAD models and testing of new CAD models.
- variables used for tuning the number of hidden layers and neurons.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.
- activation function for hidden layers: ReLu.

neuralNetwork_CrossValidation.m - The code to build and perform k-fold cross-validation on predictive model using feedforward-backpropagation neural network.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- perform k-fold cross validation with validation results of trained CAD models.
- variables used for tuning the number of hidden layers and neurons.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.
- activation function for hidden layers: ReLu.

neuralNetwork_architectureTesting.m - The code to build multiple predictive models using different architecture of feedforward-backpropagation neural network to tune the number of hidden layers and neurons.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- tuning the number of hidden layers and neurons using validation results of trained CAD models.
- variables used for tuning the number of hidden layers and neurons.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.
- activation function for hidden layers: ReLu.

polynomialRegression_Final.m - The latest code to build predictive model with Polynomial Regression to estimate printing time and material usage for trained and untrained CAD models at new printing parameters.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- predict printing time and material usage for validation of trained CAD models and testing of new CAD models.
- 3rd polynomial for best performance.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.

polynomialRegression_CrossValidation.m - The code to build and perform k-fold cross-validation on predictive model using Polynomial Regression.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- perform k-fold cross validation with validation results of trained CAD models.
- 3rd polynomial for best performance.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.

polynomialRegression_featureTesting.m - The code to build two predictive models using different sets of input features with Polynomial Regression to compare the results of different inputs.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.  
Input: spreadsheets located in Data\Final Data - Processed.xlsx  
SUMMARY:

- comparing the performance of predictive models with different sets of input feature
- tuning the degree of polynomial with variable.
- loss function: Mean Squared Percentage Error.
- perfomance metrics: R-Squared and Mean Absolute Percentage Error.

svmRegression.m - The latest code to build predictive model using Support Vector Machine with different kernels.  
Histogram, Actual vs Predicted and Residual Analysis plots are used for visualisation.
