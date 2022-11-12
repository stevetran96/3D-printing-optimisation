% neuralNetwork_CrossValidation.m: This script constructs a feedforward
% -backpropagation neural network to predict printing time and material 
% usage for new CAD models at new printing parameters.
% The number of layers for the neural network is a variable and can be 
% tuned. The number of neurons for each layer is the same and can be tuned. 
% k-fold cross validation was use to evaluate the performance of the 
% algorithm.The performance metric used were R-Squared and Mean Absolute 
% Percentage Error.
% The loss function used was Mean Squared Percentage Error.
% The activation function used for the hidden layers was ReLu function. 
% The activation function for the output layer was a linear function.
% 
% Author: Steve Tran                           
% Date created: 5/9/2019
clc;
clear;

%% Import traning data
totalInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
totalOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

featureNumber = length(totalInput(1,:));
totalInput = [ones(length(totalOutput),1), totalInput];

% Setting the number of k-fold
k = 10;
range = floor(length(totalOutput)/k);
index = 1;

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
end

for f=1:k
    trainInput = fold(f).trainInput;
    trainOutput = fold(f).trainOutput;
    trainSize = fold(f).trainSize;
    
    testInput = fold(f).testInput;
    testOutput = fold(f).testOutput;
    testSize = fold(f).testSize;
    
    %% Initialise neural network
    hiddenLayers = 2;
    hiddenNeurons = 30;
    layer(1).theta = rand(hiddenNeurons,featureNumber+1)*2-1;   % input layer - hidden layer weight
    for i=2:hiddenLayers
        layer(i).theta = rand(hiddenNeurons,hiddenNeurons+1)*2-1;
    end
    layer(hiddenLayers+1).theta = rand(1,hiddenNeurons+1)*2-1;               % hidden layer - output layer weight
    thetaVec = layer(1).theta(:);
    for i=2:(hiddenLayers+1)
        thetaVec = [thetaVec; layer(i).theta(:)];
    end

    %% Optimization using fminunc
    options = optimset('PlotFcns','optimplotfval','GradObj','on','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
    [optTheta, functionVal, exitFlag, output] = fminunc(@(thetaVec) reluCost(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureNumber+1),thetaVec,options);
    % gradCheck = gradientChecking(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureNumber+1);

    % Prediction for the current fold
    [~,~,fold(f).trainPredicted] = reluCost(optTheta,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureNumber+1);
    [~,~,fold(f).testPredicted] = reluCost(optTheta,testInput,testOutput,testSize,hiddenLayers,hiddenNeurons,featureNumber+1);
    
    % Evaluating the performance of the model at the current fold
    [fold(f).trainMAPE,fold(f).trainRsq,fold(f).trainAdjRsq] = performanceMetric(fold(f).trainPredicted,trainOutput,trainSize, featureNumber);
    [fold(f).testMAPE,fold(f).testRsq,fold(f).testAdjRsq] = performanceMetric(fold(f).testPredicted,testOutput,testSize, featureNumber);

    tempPlot = linspace(0,100000,1000);
    xAxis = linspace(-50000,50000,1000);
    yAxis = zeros(1,length(xAxis));

    figure
    subplot(1,3,1)
    plot(fold(f).testPredicted,testOutput,'kx',tempPlot,tempPlot,'b');
    axis([min(fold(f).testPredicted)-100 max(fold(f).testPredicted)+100 min(testOutput)-100 max(testOutput)+100])
    legend({'Data points','Actual = Predicted'},'Location','southeast');
    xlabel('Predicted');
    ylabel('Actual');
    title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Actual vs Predicted (in minutes)");

    subplot(1,3,2)
    histogram(100*(testOutput - fold(f).testPredicted)./testOutput);
    xlabel('Error = 100*(Target-Output)/Target (%)');
    ylabel('Number of instances');
    title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Percentage Error Histogram");

    subplot(1,3,3)
    plot(fold(f).testPredicted,testOutput-fold(f).testPredicted,'.b',xAxis,yAxis,'-k');
    axis([min(fold(f).testPredicted)-100 max(fold(f).testPredicted)+100 min(testOutput-fold(f).testPredicted)-100 max(testOutput-fold(f).testPredicted)+100])
    xlabel('Outputs');
    ylabel('Residuals');
    title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Residual Analysis");
end

% Calculate the average of performance metrics
results.trainMAPE = 0;
results.trainRsq = 0;
results.trainAdjRsq = 0;
results.testMAPE = 0;
results.testRsq = 0;
results.testAdjRsq = 0;
for f=1:k
    results.trainMAPE = results.trainMAPE + fold(f).trainMAPE;
    results.trainRsq = results.trainRsq + fold(f).trainRsq;
	results.trainAdjRsq = results.trainAdjRsq + fold(f).trainAdjRsq;
    results.testMAPE = results.testMAPE + fold(f).testMAPE;
    results.testRsq = results.testRsq + fold(f).testRsq;
    results.testAdjRsq = results.testAdjRsq + fold(f).testAdjRsq;
end
results.trainMAPE = results.trainMAPE/k;
results.trainRsq = results.trainRsq/k;
results.trainAdjRsq = results.trainAdjRsq/k;
results.testMAPE = results.testMAPE/k;
results.testRsq = results.testRsq/k;
results.testAdjRsq = results.testAdjRsq/k;

%% Matrix implementation of ReLU activation function
function [costJ, gradientJ, predictedOutput] = reluCost(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureSize)
    % Reshape theta matrices
    indexStart = 1;
    indexEnd = hiddenNeurons*featureSize;
    trainLayer(1).theta = reshape(thetaVec(indexStart:indexEnd),hiddenNeurons,featureSize);
    for i=2:hiddenLayers
        indexStart = indexEnd + 1;
        indexEnd = indexStart + hiddenNeurons*(hiddenNeurons+1)-1;
        trainLayer(i).theta = reshape(thetaVec(indexStart:indexEnd),hiddenNeurons,hiddenNeurons+1);
    end
    [~,noLayer] = size(trainLayer);
    indexStart = indexEnd + 1;
    indexEnd = indexStart + hiddenNeurons;
    trainLayer(noLayer+1).theta = reshape(thetaVec(indexStart:indexEnd),1,hiddenNeurons+1);
    [~,noLayer] = size(trainLayer);
    
    % Feedforward
    trainLayer(1).input = transpose(trainInput); % input layer activation value
    trainLayer(1).output = trainLayer(1).input;
    
    % Activation function for hidden layer
    for i=1:hiddenLayers
        trainLayer(i+1).input = trainLayer(i).theta*trainLayer(i).output;
        [row,col] = size(trainLayer(i+1).input);
        for r=1:row
            for c=1:col
                trainLayer(i+1).output(r,c) = max(0,trainLayer(i+1).input(r,c)); % calculate hidden layer activation value a
            end
        end
        trainLayer(i+1).output = [ones(1,trainSize);trainLayer(i+1).output];
    end
    
    % Activation function for output layer
    
    trainLayer(noLayer+1).input = trainLayer(noLayer).theta*trainLayer(noLayer).output;
    trainLayer(noLayer+1).output = transpose(trainLayer(noLayer+1).input);
    
    costJ = (10000/(2*trainSize))*(sum(((trainLayer(noLayer+1).output - trainOutput)./trainOutput).^2));
    predictedOutput = trainLayer(noLayer+1).output;
    
    % Backpropagation
    trainLayer(noLayer+1).delta = (trainLayer(noLayer+1).output - trainOutput)./(trainOutput.^2);
    
    for i=noLayer:-1:1
        [row,col] = size(trainLayer(i).output);
        for r=1:row
            for c=1:col
                if (trainLayer(i).output(r,c) > 0)
                    trainLayer(i).dOdI(r,c) = 1; 
                else
                    trainLayer(i).dOdI(r,c) = 0;
                end
            end
        end
        trainLayer(i).delta = trainLayer(i+1).delta * trainLayer(i).theta .* transpose(trainLayer(i).dOdI);
        trainLayer(i).delta(:,1) = [];
        
        trainLayer(i).gradient  = (10000/trainSize)*(transpose(trainLayer(i+1).delta)*transpose(trainLayer(i).output));
    end
    
    gradientJ = trainLayer(1).gradient(:);
    for i=2:noLayer
        gradientJ = [gradientJ;trainLayer(i).gradient(:)];
    end
end

%% Gradient Checking
function [gradCheck] = gradientChecking(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureSize)
    [~,testGradient] = reluCost(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureSize);
    threshold = 0.01;
    epsilon = 0.0001;
    thetaMin = thetaVec;
    thetaMax = thetaVec;
    for i=1:length(thetaVec)
        thetaMin(i) = thetaMin(i) - epsilon;
        thetaMax(i) = thetaMax(i) + epsilon;
        [testCostMinus] = reluCost(thetaMin,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureSize);
        [testCostPlus] = reluCost(thetaMax,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureSize);

        testGradientApprox(i,:) = (testCostPlus-testCostMinus)/(2*epsilon);

        thetaMin(i) = thetaMin(i) + epsilon;
        thetaMax(i) = thetaMax(i) - epsilon;
    end
    
    if (sum(abs(testGradient-testGradientApprox)) < threshold)
        gradCheck = true;
    else
        gradCheck = false;
    end
end

%% Calculat Mean Absolute Percentage Error as a performance metric
function [mape,Rsquared,adjustedRsquared] = performanceMetric(predictedOutput,output,dataSize,termNumber)
    mape = (100/dataSize)*(sum(abs(predictedOutput - output)./output));
    Rsquared = 1 - ((sum((output - predictedOutput).^2))/(sum((output - mean(output)).^2)));
    adjustedRsquared = 1 - (((1-Rsquared)*(dataSize-1))/(dataSize-termNumber-1));
end