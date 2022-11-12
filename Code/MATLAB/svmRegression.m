% svmRegression.m: This script uses MATLAB Support Vector Machine 
% Regression model to analyse the data. 
% The performance metric used were R-Squared and Mean Absolute 
% Percentage Error.
% 
% Author: Steve Tran                           
% Date created: 15/8/2019

clc;
clear;

%% Import traning data
trainInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
trainOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

% Gettiing the number of features
featureNumber = length(trainInput(1,:));

% Adding a column of ones before the input
trainInput = [ones(length(trainInput(:,1)),1), trainInput];

% Determine the validation size
validationSize = 200;
% Randomly selecting datapoints from the training set for validation
for i=1:validationSize
    temp = randi(length(trainOutput),1);
    validationInput(i,:) = trainInput(temp,:);
    validationOutput(i,:) =  trainOutput(temp);
    trainInput(temp,:) = [];
    trainOutput(temp) = [];
end

testInput = xlsread('Final Data - Processed.xlsx',4,'A2:O81');
testOutput = xlsread('Final Data - Processed.xlsx',4,'Q2:Q81');
% Adding a column of ones before the input
testInput = [ones(length(testInput(:,1)),1), testInput];
testSize = length(testOutput);

% Getting number of datapoints in training
trainSize = length(trainOutput);     

%% Fitting SVM Regression model with Gaussian kernel
% Fit the model
svmModel = fitrsvm(trainInput,trainOutput,'KernelFunction','gaussian');

% Prediction 
trainPredicted = predict(svmModel,trainInput);
validationPredicted = predict(svmModel,validationInput);
testPredicted = predict(svmModel,testInput);

% Evaluating the performance of the model
[result.trainError,result.trainRsq,result.trainAdjRsqu] = performanceMetric(trainPredicted,trainOutput,trainSize, featureNumber);
[result.validationError,result.validationRsq,result.validationAdjRsqu] = performanceMetric(validationPredicted,validationOutput,validationSize, featureNumber);
[result.testError,result.testRsq,result.testAdjRsqu] = performanceMetric(testPredicted,testOutput,testSize, featureNumber);

tempPlot = linspace(0,100000,1000);
xAxis = linspace(-50000,50000,1000);
yAxis = zeros(1,length(xAxis));

% Plotting validation graphs
figure
subplot(1,3,1)
plot(validationPredicted,validationOutput,'kx',tempPlot,tempPlot,'b');
axis([min(validationPredicted)-100 max(validationPredicted)+100 min(validationOutput)-100 max(validationOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title('SVM Regression with Gaussian Kernel Validation: Actual vs Predicted');

subplot(1,3,2)
histogram(100*(validationOutput-validationPredicted)./validationOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title('SVM: Histogram of Validation Prediction Error');

subplot(1,3,3)
plot(validationPredicted,validationOutput-validationPredicted,'.b',xAxis,yAxis,'-k');
axis([min(validationPredicted)-100 max(validationPredicted)+100 min(validationOutput-validationPredicted)-100 max(validationOutput-validationPredicted)+100])
xlabel('Outputs');
ylabel('Residuals');
title('SVM Validation Residual Analysis');

% Plotting testing graphs
figure
subplot(1,3,1)
plot(testPredicted,testOutput,'kx',tempPlot,tempPlot,'b');
axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput)-100 max(testOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title('SVM Regression with Gaussian Kernel Testing: Actual vs Predicted');

subplot(1,3,2)
histogram(100*(testOutput-testPredicted)./testOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title('SVM: Histogram of Testing Prediction Error');

subplot(1,3,3)
plot(testPredicted,testOutput-testPredicted,'.b',xAxis,yAxis,'-k');
axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput-testPredicted)-100 max(testOutput-testPredicted)+100])
xlabel('Outputs');
ylabel('Residuals');
title('SVM Testing Residual Analysis');

%% Calculat Mean Absolute Percentage Error as a performance metric
function [mape,Rsquared,adjustedRsquared] = performanceMetric(predictedOutput,output,dataSize,featureNumber)
    mape = (100/dataSize)*(sum(abs(predictedOutput - output)./output));
    Rsquared = 1 - ((sum((output - predictedOutput).^2))/(sum((output - mean(output)).^2)));
    adjustedRsquared = 1 - (((1-Rsquared)*(dataSize-1))/(dataSize-featureNumber-1));
end