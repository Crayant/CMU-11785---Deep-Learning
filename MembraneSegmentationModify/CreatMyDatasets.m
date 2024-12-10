clc
clear
close all;


% labelDir = "ac4_seg_daniel";
% targetDir = "ac4_seg_daniel_bw";
% prefix = 'ac4_daniel_s';
% r1=1;r2=100;

%%
output_size = 256;

labelDir = "ac3_EM";
targetDir = "ac3_EM_patch_256";
prefix = 'Thousand_highmag_256slices_2kcenter_1k_inv_';
r1=0;r2=255;%注意下标开始的位置
mkdir(targetDir)%新建文件夹

% labelDir = "ac4_EM";
% targetDir = "ac4_EM_patch";
% prefix = 'affinecropped4_inv_';
% r1=1;r2=100;

%处理原图成小补丁
for i = r1:r2
    s = num2str(i*1e-4, '%1.4f');
    s = s(3:end);
    s_f = strcat(prefix,s);%就是编写一个文件名
   
    
    I = imread(strcat(labelDir,'/',s_f, '.png'));%读了文件名

    %out_eval = double(evalin('base', strcat('break_image(',s_f,',256)')))/255;
    out = break_image(I, output_size);%实现把1024*1024大小的图像I转化为output_size*output_size大小的补丁，
    %(1024/output_size)*(1024/output_size)个补丁，成为output_size*output_size的多个通道所以是多张图片叠在一起
    
    for patch_j = 1: size(out,3)%size(out,3)是一张图片的补丁数量
        filename = strcat(targetDir,'/',s,'_',int2str(patch_j), '.png')%根据补丁的序号取名字
        train_data = out(:,:,patch_j);
        mask = train_data(:,:,1);
        train_data = cat(3,train_data,mask);
        train_data = cat(3,train_data,mask);
        imwrite(uint8(train_data),filename,'png');%保存补丁为图片
    end
    clear(s_f)
end

%%


labelDir = "ac3_dbseg_images";
targetDir = "ac3_dbseg_images_bw_new_256";
prefix = 'ac3_daniel_s';
r1=1;r2=256;%注意下标开始的位置
mkdir(targetDir)
%处理细胞膜的标记图成为只是细胞膜的图
for i = r1:r2
    s = num2str(i*1e-4, '%1.4f');
    s = s(3:end);

    I = imread(strcat(labelDir,'/',strcat(prefix,s), '.png'));

    s = r2 - str2num(s);%因为是逆序的
    
    s = num2str(s*1e-4, '%1.4f');
    s = s(3:end);
%     
    out = get_border(I);

%     disp(s)
    filename = strcat(targetDir,'/',  strcat(prefix,s) , '.png')
    imwrite(uint8(out)*255,filename,'png');
    
 
end
%%

labelDir = "ac3_dbseg_images_bw_new_256";
targetDir = "ac3_dbseg_images_bw_patch_new_256";
prefix = 'ac3_daniel_s';
r1=0;r2=255;
mkdir(targetDir)
%处理细胞膜图成为补丁
% labelDir = "ac4_seg_daniel_bw";
% targetDir = "ac4_seg_daniel_bw_patch";
% prefix = 'ac4_daniel_s';
% r1=0;r2=99;


for i = r1:r2

    I = imread(fullfile(labelDir,sprintf('%s%04d.png',prefix,i)));

    out = break_image(I,output_size); % break_image(eval(s_f),256);
    for patch_j = 1: size(out,3)
        filename = strcat(targetDir,'/',sprintf('%04d',i),'_',int2str(patch_j), '.png')
        imwrite(out(:,:,patch_j),filename,'png');
    end
   
end




