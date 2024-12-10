%%
clc
clear
close all;
%%


imageDir = "ac3_EM_patch_256_overlap";
labelDir = "ac3_dbseg_images_bw_patch_new_256_overlap";

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

%Create the Network
imageSize = [256 256 3];
numClasses = 2;
encoderDepth = 6;
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

%Balance Classes Using Class Weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

%Specify the class weights using a pixelClassificationLayer
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

load_name='MyNet_tv9901_epoch8.mat';
load_net=1;
if load_net==1
    load (load_name);
%     lgraph = layerGraph(net);
end

MaxEpoch = 4;
options = trainingOptions('adam','InitialLearnRate',1e-5, 'Shuffle', 'every-epoch', ...
    'MaxEpochs',MaxEpoch,'VerboseFrequency',10, 'MiniBatchSize', 4,'ExecutionEnvironment','auto',...
    'ValidationData', pximds_val, 'ValidationFrequency', 450,'Plots','training-progress');

%%

%     [net,info] = trainNetwork(pximds_train,layers,options);
%     save(strcat('deeplab', int2str(i*2),'.mat'), 'net', 'info');


for training_iter = 3:50
    if training_iter == 1
        [net,info] = trainNetwork(pximds_train,lgraph,options);
    else 
        lgraph = layerGraph(net);
        [net,info] = trainNetwork(pximds_train,lgraph,options);
    end
    %%% layers = layerGraph(net);
    save(strcat('MyNet_tv5050_no_overlap_epoch', int2str(training_iter*MaxEpoch),'.mat'), 'net', 'info');
end


