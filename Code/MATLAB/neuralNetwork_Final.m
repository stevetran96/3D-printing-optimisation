% neuralNetwork_Final.m: This script constructs a feedforward-backpropagation
% neural network to predict printing time and material usage for new CAD 
% models at new printing parameters.
% The number of layers for the neural network is a variable and can be 
% tuned. The number of neurons for each layer is the same and can be tuned. 
% The performance metric used were R-Squared and Mean Absolute Percentage 
% Error.
% The loss function used was Mean Squared Percentage Error.
% The activation function used for the hidden layers was ReLu function. 
% The activation function for the output layer was a linear function.
% 
% Author: Steve Tran                           
% Date created: 2/9/2019

clc;
clear;

%% Import traning data
trainInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2787');
trainOutput = xlsread('Final Data - Processed.xlsx',2,'P2:P2787');

featureNumber = length(trainInput(1,:));
trainInput = [ones(length(trainInput(:,1)),1), trainInput];

testInput = xlsread('Final Data - Processed.xlsx',4,'A2:O96');
testOutput = xlsread('Final Data - Processed.xlsx',4,'P2:P96');
testSize = length(testOutput);
testInput = [ones(length(testInput(:,1)),1), testInput];

% Randomly select datapoints as validation set
validationSize = 200;
for i=1:validationSize
    temp = randi(length(trainOutput),1);
    validationInput(i,:) = trainInput(temp,:);
    validationOutput(i,:) =  trainOutput(temp);
    trainInput(temp,:) = [];
    trainOutput(temp) = [];
end

trainSize = length(trainOutput);        % size of training set

%% Initialise neural network
hiddenLayers = 3;
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
gradCheck = gradientChecking(thetaVec,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureNumber+1);

% Prediction 
[~,~,trainPredicted] = reluCost(optTheta,trainInput,trainOutput,trainSize,hiddenLayers,hiddenNeurons,featureNumber+1);
[~,~,validationPredicted] = reluCost(optTheta,validationInput,validationOutput,validationSize,hiddenLayers,hiddenNeurons,featureNumber+1);
[~,~,testPredicted] = reluCost(optTheta,testInput,testOutput,testSize,hiddenLayers,hiddenNeurons,featureNumber+1);

% Evaluating the performance of the model
[result.trainError,result.trainRsq,result.trainAdjRsq] = performanceMetric(trainPredicted,trainOutput,trainSize, featureNumber);
[result.validationError,result.validationRsq,result.validationAdjRsq] = performanceMetric(validationPredicted,validationOutput,validationSize, featureNumber);
[result.testError,result.testRsq,result.testAdjRsq] = performanceMetric(testPredicted,testOutput,testSize, featureNumber);

% Plotting graphs
tempPlot = linspace(0,100000,1000);
xAxis = linspace(-50000,50000,1000);
yAxis = zeros(1,length(xAxis));

% Plotting validation results
figure
subplot(1,3,1)
plot(validationPredicted,validationOutput,'kx',tempPlot,tempPlot,'b');
axis([min(validationPredicted)-100 max(validationPredicted)+100 min(validationOutput)-100 max(validationOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Validation Actual vs Predicted (in minutes)");

subplot(1,3,2)
histogram(100*(validationOutput-validationPredicted)./validationOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Validation Percentage Error Histogram");

subplot(1,3,3)
plot(validationPredicted,validationOutput-validationPredicted,'.b',xAxis,yAxis,'-k');
axis([min(validationPredicted)-100 max(validationPredicted)+100 min(validationOutput-validationPredicted)-100 max(validationOutput-validationPredicted)+100])
xlabel('Outputs');
ylabel('Residuals');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Residual Analysis");

% Plotting testing results
figure
subplot(1,3,1)
plot(testPredicted,testOutput,'kx',tempPlot,tempPlot,'b');
axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput)-100 max(testOutput)+100])
legend({'Data points','Actual = Predicted'},'Location','southeast');
xlabel('Predicted');
ylabel('Actual');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Testing Actual vs Predicted (in minutes)");

subplot(1,3,2)
histogram(100*(testOutput-testPredicted)./testOutput,10);
xlabel('Percentage Error (%)');
ylabel('Number of instances');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Testing Percentage Error Histogram");

subplot(1,3,3)
plot(testPredicted,testOutput-testPredicted,'.b',xAxis,yAxis,'-k');
axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput-testPredicted)-100 max(testOutput-testPredicted)+100])
xlabel('Outputs');
ylabel('Residuals');
title(num2str(hiddenLayers)+" Layers "+num2str(hiddenNeurons)+" Neurons Neural Network - Residual Analysis");

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