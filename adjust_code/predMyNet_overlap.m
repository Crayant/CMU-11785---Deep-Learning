clc
clear all;
close all;
[data] = readvolume('ac4_EM');
% load ('deeplab_epoch200.mat');
% load('unet_batchsizes4_epoch25.mat')
load('MyNet_tv9901_epoch8.mat');
lgraph = layerGraph(net);

chunk_size = 256;
s = size(data);
num_pieces = s(1)/chunk_size;

Y_pred = zeros(s);
score = zeros(chunk_size, chunk_size, s(3));
% bigger_image = padarray(data, [64, 64], 0); % pad with zeros so it doesn't ecceed dimensions later in loop

% forecast
for im = 1:size(data, 3)
    im
    for j = 1:num_pieces
        for k = 1:num_pieces
            [Y_pred((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im), score((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im)] = ...
                semanticseg(double(data((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1: k*chunk_size, im)), net);
        end
    end
end

% Vertical patch
for im = 1:size(data, 3)
    im
    for j = 1:num_pieces
        for k = 1:num_pieces-1
             [result_a,result_b]=semanticseg(double(data((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1+chunk_size/2: k*chunk_size+chunk_size/2, im)), net);
            Y_pred((j-1)*chunk_size +1: j*chunk_size, (k-1)*chunk_size +1+chunk_size*3/4: k*chunk_size+chunk_size/4, im)=result_a(:,chunk_size/4+1:chunk_size*3/4);          
        end
    end
end

% Horizontal patch
offset=20;
for im = 1:size(data, 3)
    im
    for j = 1:num_pieces
        for k = 1:num_pieces-1
             [result_a,result_b]=semanticseg(double(data((k-1)*chunk_size +1+chunk_size/2: k*chunk_size+chunk_size/2,(j-1)*chunk_size +1: j*chunk_size, im)), net);
            Y_pred( (k-1)*chunk_size +1+chunk_size*3/4: k*chunk_size+chunk_size/4,(j-1)*chunk_size +1+offset: j*chunk_size-offset, im)=result_a(chunk_size/4+1:chunk_size*3/4,offset:chunk_size-offset);          
        end
    end
end
% targetDir = "ac4_seg_images_deeplab200";
targetDir = "ac4_seg_images_MyNet_Pred_batchsize4_epoch30_overlap";
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



