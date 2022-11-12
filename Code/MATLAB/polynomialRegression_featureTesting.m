% polynomialRegression_featureTesting.m: This script constructs polynomial terms and
% use polynomial regression to analyse the input data. 
% This code compares two sets of input feature with the same set of output 
% by training two predictive models. The performance of the models indicate
% how well a set of input feature can predict the output.
% The performance metric used were R-Squared and Mean Absolute 
% Percentage Error.
% The loss function used was Mean Squared Percentage Error.
% 
% Author: Steve Tran                           
% Date created: 23/8/2019

clc;
clear;

%% Import traning data
firstInput = xlsread('Final Data - Processed.xlsx',2,'A2:P2457');
firstOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

secondInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
secondOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

featureListOne = 1:length(firstInput(1,:));
featureListTwo = 1:length(secondInput(1,:));

testSize = 120;
for i=1:testSize
    temp = randi(length(firstOutput),1);
    
    testInputOne(i,:) = firstInput(temp,:);
    testOutputOne(i,:) =  firstOutput(temp);
    
    testInputTwo(i,:) = secondInput(temp,:);
    testOutputTwo(i,:) =  firstOutput(temp);
    
    firstInput(temp,:) = [];
    firstOutput(temp) = [];

    secondInput(temp,:) = [];
    secondOutput(temp) = [];
end


maxDegree = 3;
for d=1:maxDegree
    if (d==1)
        degree(d).termMatrixOne = combvec(featureListOne);
        degree(d).termMatrixTwo = combvec(featureListTwo);
    else
        degree(d).termMatrixOne = combvec(degree(d-1).termMatrixOne,featureListOne);
        degree(d).termMatrixTwo = combvec(degree(d-1).termMatrixTwo,featureListTwo);
    end
    degree(d).termListOne = unique(sort(transpose(degree(d).termMatrixOne),2), 'rows');
    degree(d).termListTwo = unique(sort(transpose(degree(d).termMatrixTwo),2), 'rows');
    
	for t=1:length(degree(d).termListOne(:,1))
        degree(d).trainOne(:,t) = ones(length(firstInput(:,1)),1);
        degree(d).testOne(:,t) = ones(length(testInputOne(:,1)),1);
        for n=1:length(degree(d).termListOne(1,:))
            degree(d).trainOne(:,t) = degree(d).trainOne(:,t) .* firstInput(:,degree(d).termListOne(t,n));
            degree(d).testOne(:,t) = degree(d).testOne(:,t) .* testInputOne(:,degree(d).termListOne(t,n));
        end
    end
    
	for t=1:length(degree(d).termListTwo(:,1))
        degree(d).trainTwo(:,t) = ones(length(secondInput(:,1)),1);
        degree(d).testTwo(:,t) = ones(length(testInputTwo(:,1)),1);
        for n=1:length(degree(d).termListTwo(1,:))
            degree(d).trainTwo(:,t) = degree(d).trainTwo(:,t) .* secondInput(:,degree(d).termListTwo(t,n));
            degree(d).testTwo(:,t) = degree(d).testTwo(:,t) .* testInputTwo(:,degree(d).termListTwo(t,n));
        end
    end
    
end

for cd=1:maxDegree
    if (cd==1)
        degree(cd).trainOne = [ones(length(firstInput(:,1)),1), degree(cd).trainOne];
        degree(cd).testOne = [ones(length(testInputOne(:,1)),1), degree(cd).testOne];
        
        degree(cd).trainTwo = [ones(length(secondInput(:,1)),1), degree(cd).trainTwo];
        degree(cd).testTwo = [ones(length(testInputTwo(:,1)),1), degree(cd).testTwo];
    else
        degree(cd).trainOne = [degree(cd-1).trainOne, degree(cd).trainOne];
        degree(cd).testOne = [degree(cd-1).testOne, degree(cd).testOne];
        
        degree(cd).trainTwo = [degree(cd-1).trainTwo , degree(cd).trainTwo];
        degree(cd).testTwo = [degree(cd-1).testTwo, degree(cd).testTwo];
    end
end

for d=3:maxDegree
    trainInputOne = degree(d).trainOne;
    trainOutputOne = firstOutput;
    trainSizeOne = length(trainOutputOne);    % number of datapoints in training
    termNumberOne = length(trainInputOne(1,:));
    
	trainInputTwo = degree(d).trainTwo;
    trainOutputTwo = secondOutput;
    trainSizeTwo = length(trainOutputTwo);    % number of datapoints in training
    termNumberTwo = length(trainInputTwo(1,:));

    %% Advanced Optimization
    lambda = 1;
    
    % Optimization using fminunc
    options = optimset('GradObj','on','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
    initThetaOne = zeros(termNumberOne,1);    % number of parameters (features+1)
    [degree(d).thetaOne] = fminunc(@(theta) linearCost(theta,trainInputOne,trainOutputOne,trainSizeOne, lambda),initThetaOne,options);
    [degree(d).trainOneError,degree(d).trainOneRsq,degree(d).trainOneAdjRsqu] = performanceMetric(degree(d).thetaOne, trainInputOne,trainOutputOne,trainSizeOne, termNumberOne-1);
    [degree(d).testOneError,degree(d).testOneRsq,degree(d).testOneAdjRsqu] = performanceMetric(degree(d).thetaOne, degree(d).testOne,testOutputOne,length(testOutputOne), termNumberOne-1);

    predictedOne = degree(d).testOne*degree(d).thetaOne;
    
    tempPlot = linspace(0,5000,100);
    xAxis = linspace(-5000,5000,100);
    yAxis = zeros(1,length(xAxis));
    
    figure
    subplot(2,4,1)
    plot(predictedOne,testOutputOne,'kx',tempPlot,tempPlot,'b');
    axis([min(predictedOne)-100 max(predictedOne)+100 min(testOutputOne)-100 max(testOutputOne)+100])
    legend({'Data points','Actual = Predicted'},'Location','southeast');
    xlabel('Predicted');
    ylabel('Actual');
    title('ONE: Actual vs Predicted (in minutes)');
    
    subplot(2,4,2)
    histogram(testOutputOne - predictedOne);
    xlabel('Error = Target - Output (mins)');
    ylabel('Number of instances');
    title('ONE: Time Error Histogram in minutes');
    
	subplot(2,4,3)
    histogram(100*(testOutputOne-predictedOne)./testOutputOne);
    xlabel('Error = 100*(Target-Output)/Target (%)');
    ylabel('Number of instances');
    title('ONE: Percentage Error Histogram');
    
    subplot(2,4,4)
    plot(predictedOne,testOutputOne-predictedOne,'.b',xAxis,yAxis,'-k');
    axis([min(predictedOne)-100 max(predictedOne)+100 min(testOutputOne-predictedOne)-100 max(testOutputOne-predictedOne)+100])
    xlabel('Outputs');
    ylabel('Residuals');
    title('ONE: Residual Analysis');
    
    
    initThetaTwo = zeros(termNumberTwo,1); 
    [degree(d).thetaTwo] = fminunc(@(newtheta) linearCost(newtheta,trainInputTwo,trainOutputTwo,trainSizeTwo, lambda),initThetaTwo,options);
    [degree(d).trainTwoError,degree(d).trainTwoRsq,degree(d).trainTwoAdjRsqu] = performanceMetric(degree(d).thetaTwo,trainInputTwo,trainOutputTwo,trainSizeTwo, termNumberTwo-1);
    [degree(d).testTwoError,degree(d).testTwoRsq,degree(d).testTwoAdjRsqu] = performanceMetric(degree(d).thetaTwo, degree(d).testTwo,testOutputTwo,length(testOutputTwo), termNumberTwo-1);

    predictedTwo = degree(d).testTwo*degree(d).thetaTwo;
    
    subplot(2,4,5)
    plot(predictedTwo,testOutputTwo,'kx',tempPlot,tempPlot,'b');
    axis([min(predictedTwo)-100 max(predictedTwo)+100 min(testOutputTwo)-100 max(testOutputTwo)+100])
    legend({'Data points','Actual = Predicted'},'Location','southeast');
    xlabel('Predicted');
    ylabel('Actual');
    title('TWO: Actual vs Predicted (in minutes)');
    
    subplot(2,4,6)
    histogram(testOutputTwo - predictedTwo);
    xlabel('Error = Target - Output (mins)');
    ylabel('Number of instances');
    title('TWO: Time Error Histogram in minutes');
    
	subplot(2,4,7)
    histogram(100*(testOutputTwo - predictedTwo)./testOutputTwo);
    xlabel('Error = 100*(Target-Output)/Target (%)');
    ylabel('Number of instances');
    title('TWO: Percentage Error Histogram');
    
    subplot(2,4,8)
    plot(predictedTwo,testOutputTwo-predictedTwo,'.b',xAxis,yAxis,'-k');
    axis([min(predictedTwo)-100 max(predictedTwo)+100 min(testOutputTwo-predictedTwo)-100 max(testOutputTwo-predictedTwo)+100])
    xlabel('Outputs');
    ylabel('Residuals');
    title('TWO: Residual Analysis');   
end
    
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

%% Calculat Mean Squared Error as a performance metric
function [mape,Rsquared,adjustedRsquared] = performanceMetric(theta,input,output,dataSize,termNumber)
    predictedOutput = input*theta;
    mape = (100/dataSize)*(sum(abs(predictedOutput - output)./output));
    Rsquared = 1 - ((sum((output - predictedOutput).^2))/(sum((output - mean(output)).^2)));
    adjustedRsquared = 1 - (((1-Rsquared)*(dataSize-1))/(dataSize-termNumber-1));
end