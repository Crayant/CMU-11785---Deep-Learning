clc
clear all;
close all;
[data] = readvolume('ac4_EM');
% load ('deeplab_epoch200.mat');
% load('unet_batchsizes4_epoch25.mat')
load('MyNet_batchsize_4_epoch30.mat');
lgraph = layerGraph(net);

chunk_size = 256;
s = size(data);
num_pieces = s(1)/chunk_size;

Y_pred = zeros(s);
score = zeros(chunk_size, chunk_size, s(3));
bigger_image = padarray(data, [64, 64], 0); % pad with zeros so it doesn't ecceed dimensions later in loop

for im = 1:size(data, 3)
    im
    for j = 1:num_pieces
        for k = 1:num_pieces
            [Y_pred((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im), score((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im)] = ...
                semanticseg(double(data((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im)), net);
        end
    end
end



% targetDir = "ac4_seg_images_deeplab200";
targetDir = "ac4_seg_images_MyNet_Pred_batchsize4_epoch30";
% prefix = 'ac3_daniel_s';
r1=1;r2=size(data, 3);
mkdir(targetDir)

Y_pred(Y_pred==1)=255;
Y_pred(Y_pred==2)=0;
Y_pred_a=Y_pred;
% adjust=3;
% Y_pred_a=imerode(Y_pred,ones(adjust,adjust));
% Y_pred_a=imdilate(Y_pred_a,ones(adjust,adjust));

for i = r1:r2

    filename = strcat(targetDir,'/',sprintf('%04d',i),'_', '.png')
    imwrite(Y_pred_a(:,:,i),filename,'png');

end



