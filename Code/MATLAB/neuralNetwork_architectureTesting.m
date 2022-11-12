% neuralNetwork_architectureTesting.m: This script constructs feedforward
% -backpropagation neural networks with different architecture to tune the 
% number of hidden layers and the number of neurons. 
% The maximum number of hidden layers of the neural network and the maximum
% number of neurons for each layer are variables and can be tuned. 
% The performance of the networks for predicting printing time and material
% usage for trained CAD models at new printing parameters is used to
% evaluate the network architecture.
% The performance metric used were R-Squared and Mean Absolute Percentage 
% Error.
% The loss function used was Mean Squared Percentage Error.
% The activation function used for the hidden layers was ReLu function. 
% The activation function for the output layer was a linear function.
% 
% Author: Steve Tran                           
% Date created: 28/8/2019

clc;
clear;

%% Import traning data
trainInput = xlsread('Final Data - Processed.xlsx',2,'A2:O2457');
trainOutput = xlsread('Final Data - Processed.xlsx',2,'Q2:Q2457');

featureNumber = length(trainInput(1,:));
trainInput = [ones(length(trainInput(:,1)),1), trainInput];

% Randomly select datapoints as testing set
testSize = 120;
for i=1:testSize
    temp = randi(length(trainOutput),1);
    testInput(i,:) = trainInput(temp,:);
    testOutput(i,:) =  trainOutput(temp);
    trainInput(temp,:) = [];
    trainOutput(temp) = [];
end

trainSize = length(trainOutput);        % size of training set

%% Initialise neural network
hiddenLayers = 3;
hiddenNeurons = 30;
for l=1:hiddenLayers
    for n=10:10:hiddenNeurons
        layer(1).theta = rand(n,featureNumber+1)*2-1;   % input layer - hidden layer weight
        for i=2:l
            layer(i).theta = rand(n,n+1)*2-1;
        end
        layer(l+1).theta = rand(1,n+1)*2-1;               % hidden layer - output layer weight
        thetaVec = layer(1).theta(:);
        for i=2:(l+1)
            thetaVec = [thetaVec; layer(i).theta(:)];
        end

        %% Optimization using fminunc
        options = optimset('PlotFcns','optimplotfval','GradObj','on','MaxFunEvals',1000000000,'MaxIter',1000000000,'TolFun',0);
        [optTheta, functionVal, exitFlag, output] = fminunc(@(thetaVec) reluCost(thetaVec,trainInput,trainOutput,trainSize,l,n,featureNumber+1),thetaVec,options);
        gradCheck = gradientChecking(thetaVec,trainInput,trainOutput,trainSize,l,n,featureNumber+1);
        
        % Prediction 
        [~,~,trainPredicted] = reluCost(optTheta,trainInput,trainOutput,trainSize,l,n,featureNumber+1);
        [~,~,testPredicted] = reluCost(optTheta,testInput,testOutput,testSize,l,n,featureNumber+1);

        % Evaluating the performance of the model
        [result(l).trainError(n/10),result(l).trainRsq(n/10),result(l).trainAdjRsqu(n/10)] = performanceMetric(trainPredicted,trainOutput,trainSize, featureNumber);
        [result(l).testError(n/10),result(l).testRsq(n/10),result(l).testAdjRsqu(n/10)] = performanceMetric(testPredicted,testOutput,testSize, featureNumber);

        % Plotting graphs
        tempPlot = linspace(0,100000,1000);
        xAxis = linspace(-50000,50000,1000);
        yAxis = zeros(1,length(xAxis));

        figure
        subplot(1,3,1)
        plot(testPredicted,testOutput,'kx',tempPlot,tempPlot,'b');
        axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput)-100 max(testOutput)+100])
        legend({'Data points','Actual = Predicted'},'Location','southeast');
        xlabel('Predicted');
        ylabel('Actual');
        title(num2str(l)+" Layers "+num2str(n)+" Neurons Neural Network - Target vs Output (in minutes)");

        subplot(1,3,2)
        histogram(100*(testOutput-testPredicted)./testOutput,10);
        xlabel('Percentage Error (%)');
        ylabel('Number of instances');
        title(num2str(l)+" Layers "+num2str(n)+" Neurons Neural Network - Percentage Error Histogram");

        subplot(1,3,3)
        plot(testPredicted,testOutput-testPredicted,'.b',xAxis,yAxis,'-k');
        axis([min(testPredicted)-100 max(testPredicted)+100 min(testOutput-testPredicted)-100 max(testOutput-testPredicted)+100])
        xlabel('Outputs');
        ylabel('Residuals');
        title(num2str(l)+" Layers "+num2str(n)+" Neurons Neural Network - Residual Analysis");
    end
end

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
    
    if (sum(abs(testGradient-testGradientApprox)) < 0.0001)
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