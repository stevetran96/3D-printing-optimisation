% polynomialRegression_Final.m: This script constructs polynomial terms and
% use polynomial regression to analyse the input data and build predictive
% model.
% The model is used for validation and testing of new data.
% The performance metric used were R-Squared and Mean Absolute 
% Percentage Error.
% The loss function used was Mean Squared Percentage Error.
% 
% Author: Steve Tran                           
% Date created: 30/8/2019

clear;
clc;

% Import traning data
totalInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
totalOutput = xlsread('Final Data - Processed.xlsx',2,'S2:S2457');

featureList = 1:length(totalInput(1,:));

% Import testing data
testInput = xlsread('Final Data - Processed.xlsx',4,'A2:O81');
testOutput = xlsread('Final Data - Processed.xlsx',4,'S2:S81');
testSize = length(testOutput);

validationSize = 100;
for i=1:validationSize
    temp = randi(length(totalOutput),1);
    validationInput(i,:) = totalInput(temp,:);
    validationOutput(i,:) =  totalOutput(temp);
    totalInput(temp,:) = [];
    totalOutput(temp) = [];
end

% Setting the polynomial degree
maxDegree = 3;
% Constructing higher order polynomial terms
for d=1:maxDegree
    % Obtain the list of terms for the polynomial
    if (d==1)
        degree(d).termMatrix = combvec(featureList);
    else
        degree(d).termMatrix = combvec(degree(d-1).termMatrix,featureList);
    end
    degree(d).termList = unique(sort(transpose(degree(d).termMatrix),2), 'rows');
    
    % Calculating the terms based on the term list for the polynomial
	for t=1:length(degree(d).termList(:,1))
        degree(d).train(:,t) = ones(length(totalInput(:,1)),1);
        degree(d).validation(:,t) = ones(length(validationInput(:,1)),1);
        degree(d).test(:,t) = ones(length(testInput(:,1)),1);
        for n=1:length(degree(d).termList(1,:))
            degree(d).train(:,t) = degree(d).train(:,t) .* totalInput(:,degree(d).termList(t,n));
            degree(d).validation(:,t) = degree(d).validation(:,t) .* validationInput(:,degree(d).termList(t,n));
            degree(d).test(:,t) = degree(d).test(:,t) .* testInput(:,degree(d).termList(t,n));
        end
    end
end

% Combining the terms for each degree to obtain the complete polynomial
for cd=1:maxDegree
    if (cd==1)
        degree(cd).train = [ones(length(totalInput(:,1)),1), degree(cd).train];
        degree(cd).validation = [ones(length(validationInput(:,1)),1), degree(cd).validation];
        degree(cd).test = [ones(length(testInput(:,1)),1), degree(cd).test];
    else
        degree(cd).train = [degree(cd-1).train, degree(cd).train];
        degree(cd).validation = [degree(cd-1).validation, degree(cd).validation];
        degree(cd).test = [degree(cd-1).test, degree(cd).test];
    end
end

%% Training the model
% Initialising input and output
trainInput = degree(maxDegree).train;
trainOutput = totalOutput;
trainSize = length(trainOutput);    
termNumber = length(trainInput(1,:));

lambda = 1;

% Optimization using fminunc
options = optimset('PlotFcns','optimplotfval','GradObj','on','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
initTheta = zeros(termNumber,1);    % number of parameters (features+1)
[degree(maxDegree).optTheta] = fminunc(@(theta) linearCost(theta,trainInput,trainOutput,trainSize, lambda),initTheta,options);
% Evaluating the performance of models
[result.trainError,result.trainRsq,result.trainAdjRsqu] = performanceMetric(degree(maxDegree).optTheta, trainInput,trainOutput,trainSize, termNumber-1);
[result.validationError,result.validationRsq,result.validationAdjRsqu] = performanceMetric(degree(maxDegree).optTheta, degree(maxDegree).validation,validationOutput,validationSize, termNumber-1);
[result.testError,result.testRsq,result.testAdjRsqu] = performanceMetric(degree(maxDegree).optTheta, degree(maxDegree).test,testOutput,testSize, termNumber-1);

% Using the trained model for prediction
predictedValidation = degree(maxDegree).validation*degree(maxDegree).optTheta;
predictedTest = degree(maxDegree).test*degree(maxDegree).optTheta;

tempPlot = linspace(0,100000,1000);
xAxis = linspace(-50000,50000,1000);
yAxis = zeros(1,length(xAxis));

% Plotting graphs for validation
figure
subplot(1,3,1)
plot(predictedValidation,validationOutput,'kx',tempPlot,tempPlot,'b');
axis([min(predictedValidation)-100 max(predictedValidation)+100 min(validationOutput)-100 max(validationOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title('Polynomial Regression Validation: Actual vs Predicted');

subplot(1,3,2)
histogram(100*(validationOutput-predictedValidation)./validationOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title('Polynomial Regression: Histogram of Validation Prediction Error');

subplot(1,3,3)
plot(predictedValidation,validationOutput-predictedValidation,'.b',xAxis,yAxis,'-k');
axis([min(predictedValidation)-100 max(predictedValidation)+100 min(validationOutput-predictedValidation)-100 max(validationOutput-predictedValidation)+100])
xlabel('Outputs');
ylabel('Residuals');
title('Polynomial Regression: Validation Residual Analysis');

% Plotting graphs for testing
figure
subplot(1,3,1)
plot(predictedTest,testOutput,'kx',tempPlot,tempPlot,'b');
axis([min(predictedTest)-100 max(predictedTest)+100 min(testOutput)-100 max(testOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title('Polynomial Regression Testing: Actual vs Predicted');

subplot(1,3,2)
histogram(100*(testOutput-predictedTest)./testOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title('Polynomial Regression: Histogram of Testing Prediction Error');

subplot(1,3,3)
plot(predictedTest,testOutput-predictedTest,'.b',xAxis,yAxis,'-k');
axis([min(predictedTest)-100 max(predictedTest)+100 min(testOutput-predictedTest)-100 max(testOutput-predictedTest)+100])
xlabel('Outputs');
ylabel('Residuals');
title('Polynomial Regression: Testing Residual Analysis');

    
%% Learning Curve
% totalSet = [totalInput,totalOutput];
% optionsLearning = optimset('GradObj','off','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
% figure
% hold on
% for m=5:(length(totalInput(:,1))/10)
%     size = m*10;
%     learningSet(m).data = datasample(totalSet,size,'Replace',false);
%     learningSet(m).input = learningSet(m).data(:,1:14);
%     learningSet(m).output = learningSet(m).data(:,15);
%     [learningSet(m).theta, learningSet(m).trainCost] = fminunc(@(theta) linearCost(theta,learningSet(m).input,learningSet(m).output,length(learningSet(m).output), lambda),initTheta,optionsLearning);
%     [learningSet(m).trainError,learningSet(m).trainRsq,learningSet(m).trainAdjRsqu] = performanceMetric(learningSet(m).theta,learningSet(m).input,learningSet(m).output,length(learningSet(m).output), featureNumber);
%     [learningSet(m).testError,learningSet(m).testRsq,learningSet(m).testAdjRsqu] = performanceMetric(learningSet(m).theta,testInput,testOutput,testSize, featureNumber);
%     plot(size,learningSet(m).trainError,'rx',size,learningSet(m).testError,'bo')
% end
% legend('Training Error','Testing Error');
% xlabel('Training Set size (m)');
% ylabel('Mean Squared Error');
% title('Change in MSE versus Training Set Size');

%% Calculate cost function J and gradient dJ
%  Input: theta = j-dimensional vector of parameter theta corresponding
%         with number of features
%  Output: costJ = cost function at theta
%          gradientJ = gradient (rate of change) in cost function at theta
function [costJ, gradientJ] = linearCost(theta,input,output,dataSize,lambda)
    costJ = (10000/(2*dataSize))*(sum(((input*theta - output)./output).^2));
    gradientJ = (10000/dataSize)*transpose(input)*((input*theta - output)./(output.^2));
end

%% Calculat Mean Absolute Percentage Error as a performance metric
function [mape,Rsquared,adjustedRsquared] = performanceMetric(theta,input,output,dataSize,termNumber)
    predictedOutput = input*theta;
    mape = (100/dataSize)*(sum(abs(predictedOutput - output)./output));
    Rsquared = 1 - ((sum((output - predictedOutput).^2))/(sum((output - mean(output)).^2)));
    adjustedRsquared = 1 - (((1-Rsquared)*(dataSize-1))/(dataSize-termNumber-1));
end