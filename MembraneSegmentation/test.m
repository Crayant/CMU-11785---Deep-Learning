
labelDir = "ac3_dbseg_images_bw_new_128";
targetDir = "ac3_dbseg_images_bw_patch_new_128";
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
        filename = strcat(targetDir,'/',sprintf('%s%04d.png',prefix,i),'_',int2str(patch_j), '.png')
        imwrite(out(:,:,patch_j),filename,'png');
    end
   
end

