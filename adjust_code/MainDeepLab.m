%%
clear
close all;
clc

%%

imageDir = "ac3_EM_patch_256";
labelDir = "ac3_dbseg_images_bw_patch_new_256";

% dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
% imageDir = fullfile(dataSetDir,'trainingImages');
% labelDir = fullfile(dataSetDir,'trainingLabels');


imds = imageDatastore(imageDir);



classNames = ["border","no_border"];
labelIDs   = [255 0];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

ds = pixelLabelImageDatastore(imds,pxds);

[train, val, test] = dividerand(ds.NumObservations, 0.99, 0.01, 0);

pximds_train = partitionByIndex(ds,train);
pximds_val = partitionByIndex(ds,val);
pximds_test = partitionByIndex(ds,test);




tbl = countEachLabel(pximds_train);

% give more weight to the underrepresented class
numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / numberPixels;
classWeights = 1 ./ frequency;


% define the network
inputSize =  [256 256 1];
filterSize = 3;
numFilters = 32;
numClasses = numel(classNames);

% structure of the network suggested by deeplab
% layers = [
%     imageInputLayer(inputSize)
%     
%     convolution2dLayer(filterSize,numFilters,'DilationFactor',1,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(filterSize,numFilters,'DilationFactor',2,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(filterSize,numFilters,'DilationFactor',4,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(1,numClasses)
%     softmaxLayer
%     pixelClassificationLayer('Classes',classNames,'ClassWeights',classWeights)];

% structure of the network imitating Lu's net
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',1,'Padding','same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',2,'Padding','same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',4,'Padding','same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',4,'Padding','same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(1,numClasses)
    softmaxLayer
    pixelClassificationLayer('Classes',classNames,'ClassWeights',classWeights)];

%%


% options = trainingOptions('adam','InitialLearnRate',1e-3, 'Shuffle', 'every-epoch', ...
%     'MaxEpochs',4,'VerboseFrequency',10, 'MiniBatchSize', 4,'ExecutionEnvironment','gpu',...
%     'Plots', 'training-progress', 'ValidationData', pximds_val, 'ValidationFrequency', 500);
% 
% 
% 
% %%
% %net = trainNetwork(ds,lgraph,options)
% 
% doTraining = true; 
% if doTraining     
%     [net,info] = trainNetwork(pximds_train,lgraph,options); 
% %     [net,info] = trainNetwork(pximds_train,net.Layers,options); 
% else 
%     load(fullfile(imageDir,'trainedUnet','multispectralUnet.mat'));
% end

%%
MaxEpoch = 4;
options = trainingOptions('adam','InitialLearnRate',1e-3, 'Shuffle', 'every-epoch', ...
    'MaxEpochs',MaxEpoch,'VerboseFrequency',10, 'MiniBatchSize', 4,'ExecutionEnvironment','auto',...
    'ValidationData', pximds_val, 'ValidationFrequency', 450,'Plots','training-progress');

%%

%     [net,info] = trainNetwork(pximds_train,layers,options);
%     save(strcat('deeplab', int2str(i*2),'.mat'), 'net', 'info');


for training_iter = 1:50
    if training_iter == 1
        [net,info] = trainNetwork(pximds_train,layers,options);
    else 
        [net,info] = trainNetwork(pximds_train,net.Layers,options);
    end
    %%% layers = layerGraph(net);
    save(strcat('deeplab_tv9901_no_overlap_epoch', int2str(training_iter*MaxEpoch),'.mat'), 'net', 'info');
end


