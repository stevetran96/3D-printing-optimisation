% polynomialRegression_CrossValidation.m: This script constructs polynomial
% terms and use polynomial regression to analyse the input data. 
% k-fold cross validation was use to evaluate the performance of the 
% algorithm. 
% The results of k-fold cross validation are averaged for the number of
% folds.
% The performance metric used were R-Squared and Mean Absolute 
% Percentage Error.
% The loss function used was Mean Squared Percentage Error.
% 
% Author: Steve Tran                           
% Date created: 30/8/2019

clear;
clc;

%% Import traning data
totalInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
totalOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

featureList = 1:length(totalInput(1,:));

% Setting the number of k-fold
k = 10;
range = floor(length(totalOutput)/k);
index = 1;
% Separating the randomised data into k-fold
for f=1:k
    if (f == k)
        fold(f).trainInput = totalInput(1:(index-1),:);
        fold(f).trainOutput = totalOutput(1:(index-1),:);
        fold(f).testInput = totalInput((index:end),:);
        fold(f).testOutput = totalOutput((index:end),:);
        fold(f).trainSize = length(fold(f).trainOutput);
        fold(f).testSize = length(fold(f).testOutput);
    else
        fold(f).trainInput = [totalInput(1:(index-1),:);totalInput((index+range):end,:)];
        fold(f).trainOutput = [totalOutput(1:(index-1),:);totalOutput((index+range):end,:)];
        fold(f).testInput = totalInput(index:(index+range-1),:);
        fold(f).testOutput = totalOutput(index:(index+range-1),:);
        fold(f).trainSize = length(fold(f).trainOutput);
        fold(f).testSize = length(fold(f).testOutput);
        index = index + range;        
    end
    
    fold(f).trainTerm = ones(fold(f).trainSize,1);
    fold(f).testTerm = ones(fold(f).testSize,1);
end

% Constructing higher order polynomial terms
maxDegree = 3;
for d=1:maxDegree
    if (d==1)
        degree(d).termMatrix = combvec(featureList);
    else
        degree(d).termMatrix = combvec(degree(d-1).termMatrix,featureList);
    end
    degree(d).termList = unique(sort(transpose(degree(d).termMatrix),2), 'rows');
    
    for f=1:k
        for t=1:length(degree(d).termList(:,1))
            fold(f).tempTrain(:,t) = ones(fold(f).trainSize,1);
            fold(f).tempTest(:,t) = ones(fold(f).testSize,1);
            for n=1:length(degree(d).termList(1,:))
                fold(f).tempTrain(:,t) = fold(f).tempTrain(:,t) .* fold(f).trainInput(:,degree(d).termList(t,n));
                fold(f).tempTest(:,t) = fold(f).tempTest(:,t) .* fold(f).testInput(:,degree(d).termList(t,n));
            end
        end
        fold(f).trainTerm = [fold(f).trainTerm,fold(f).tempTrain];
        fold(f).testTerm = [fold(f).testTerm,fold(f).tempTest];
    end
end

for f=1:k
    % Initialise inputs and outputs for the current fold
    trainInput = fold(f).trainTerm;
    trainOutput = fold(f).trainOutput;
    trainSize = fold(f).trainSize;   
    
    testInput = fold(f).testTerm;
    testOutput = fold(f).testOutput;
    testSize = fold(f).testSize;
    
    termNumber = length(trainInput(1,:));

    %% Advanced Optimization

    lambda = 1;
    
    % Optimization using fminunc
    options = optimset('PlotFcns','optimplotfval','GradObj','on','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
    initTheta = zeros(termNumber,1);    % number of parameters (features+1)
    [fold(f).optTheta] = fminunc(@(theta) linearCost(theta,trainInput,trainOutput,trainSize, lambda),initTheta,options);
    [fold(f).trainMAPE,fold(f).trainRsq,fold(f).trainAdjRsqu] = performanceMetric(fold(f).optTheta, trainInput,trainOutput,trainSize, termNumber-1);
    [fold(f).testMAPE,fold(f).testRsq,fold(f).testAdjRsqu] = performanceMetric(fold(f).optTheta, testInput,testOutput,testSize, termNumber-1);

    predictedTest = testInput*fold(f).optTheta;
    
    tempPlot = linspace(0,100000,1000);
    xAxis = linspace(-50000,50000,1000);
    yAxis = zeros(1,length(xAxis));
    
    figure
    subplot(1,3,1)
    plot(predictedTest,testOutput,'kx',tempPlot,tempPlot,'b');
    axis([min(predictedTest)-100 max(predictedTest)+100 min(testOutput)-100 max(testOutput)+100])
    legend({'Data points','Actual = Predicted'},'Location','southeast');
    xlabel('Predicted');
    ylabel('Actual');
    title("Fold "+ num2str(f) +" Polynomial Regression Testing: Actual vs Predicted");
    
	subplot(1,3,2)
    histogram(100*(testOutput-predictedTest)./testOutput,10);
    xlabel('Percentage Error (%)');
    ylabel('Number of instances');
    title("Fold "+ num2str(f) +" Polynomial Regression: Histogram of Testing Prediction Error");
    
    subplot(1,3,3)
    plot(predictedTest,testOutput-predictedTest,'.b',xAxis,yAxis,'-k');
    axis([min(predictedTest)-100 max(predictedTest)+100 min(testOutput-predictedTest)-100 max(testOutput-predictedTest)+100])
    xlabel('Outputs');
    ylabel('Residuals');
    title("Fold "+ num2str(f) +" Polynomial Regression: Testing Residual Analysis");
end

% Calculate the average of performance metrics
results.trainMAPE = 0;
results.trainRsq = 0;
results.trainAdjRsqu = 0;
results.testMAPE = 0;
results.testRsq = 0;
results.testAdjRsqu = 0;
for f=1:k
    results.trainMAPE = results.trainMAPE + fold(f).trainMAPE;
    results.trainRsq = results.trainRsq + fold(f).trainRsq;
	results.trainAdjRsqu = results.trainAdjRsqu + fold(f).trainAdjRsqu;
    results.testMAPE = results.testMAPE + fold(f).testMAPE;
    results.testRsq = results.testRsq + fold(f).testRsq;
    results.testAdjRsqu = results.testAdjRsqu + fold(f).testAdjRsqu;
end
results.trainMAPE = results.trainMAPE/k;
results.trainRsq = results.trainRsq/k;
results.trainAdjRsqu = results.trainAdjRsqu/k;
results.testMAPE = results.testMAPE/k;
results.testRsq = results.testRsq/k;
results.testAdjRsqu = results.testAdjRsqu/k;

%% Calculate cost function J and gradient dJ
%  Input: theta = j-dimensional vector of parameter theta corresponding
%         with number of features
%  Output: costJ = cost function at theta
%          gradientJ = gradient (rate of change) in cost function at theta
function [costJ, gradientJ] = linearCost(theta,input,output,dataSize,lambda)
    costJ = (10000/(2*dataSize))*(sum(((input*theta - output)./output).^2));
    gradientJ = (10000/dataSize)*transpose(input)*((input*theta - output)./(output.^2));
end

%% Calculat Mean Squared Error as a performance metric
function [mape,Rsquared,adjustedRsquared] = performanceMetric(theta,input,output,dataSize,termNumber)
    predictedOutput = input*theta;
    mape = (100/dataSize)*(sum(abs(predictedOutput - output)./output));
    Rsquared = 1 - ((sum((output - predictedOutput).^2))/(sum((output - mean(output)).^2)));
    adjustedRsquared = 1 - (((1-Rsquared)*(dataSize-1))/(dataSize-termNumber-1));
end