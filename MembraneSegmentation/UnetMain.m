%%
load_net= 0;

output_size = 256;

imageSize = [output_size, output_size, 1];
% imageSize = [32 32];
numClasses = 2;
encoderDepth = 5;

if load_net==1
    load ('unet_batchsizes4_epoch25.mat');
    lgraph = layerGraph(net);
else
    %lgraph = segnetLayers(imageSize,numClasses,encoderDepth);
    lgraph = unetLayers (imageSize, numClasses, 'EncoderDepth' , encoderDepth);
    
end


plot (lgraph)

%%

imageDir = "ac3_EM_patch_256";%'kasturi_em'; %"ac3_EM_patch";
labelDir = "ac3_dbseg_images_bw_patch_new_256"%;'kasturi_labels'; %"ac3_dbseg_images_bw_patch_new";

% dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
% imageDir = fullfile(dataSetDir,'trainingImages');
% labelDir = fullfile(dataSetDir,'trainingLabels');


imds = imageDatastore(imageDir);



classNames = ["border","no_border"];
labelIDs   = [255 0];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

ds = pixelLabelImageDatastore(imds,pxds);

if load_net ==1
    load('data_split.mat');
else
    [train, val, test] = dividerand(ds.NumObservations, 0.8, 0.2, 0);
end

pximds_train = partitionByIndex(ds,train);
pximds_val = partitionByIndex(ds,val);
pximds_test = partitionByIndex(ds,test);




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
options = trainingOptions('adam','InitialLearnRate',1e-3, 'Shuffle', 'every-epoch', ...
    'MaxEpochs',5,'VerboseFrequency',10, 'MiniBatchSize', 4,'ExecutionEnvironment','auto',...
    'ValidationData', pximds_val, 'ValidationFrequency', 450);

%%
% for training_iter = 1:5
%     if training_iter == 1
%         [net,info] = trainNetwork(pximds_train,lgraph,options);
%     else 
%         [net,info] = trainNetwork(pximds_train,net.Layers,options);
%     end
%     %%% layers = layerGraph(net);
%     save(strcat('unet', int2str(training_iter*MaxEpoch),'.mat'), 'net', 'info');
% end


tic
for i = 1:5
    [net,info] = trainNetwork(pximds_train,lgraph,options);
    lgraph = layerGraph(net);
    save(strcat('unet_batchsizes4_epoch', int2str(i*4),'.mat'), 'net', 'info');
end

toc
