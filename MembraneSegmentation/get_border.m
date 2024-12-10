function BW = get_border(x)
%根据有颜色的标签获得图片的细胞膜边界
try 
    I_s = unique_rgb(x);%细胞膜的图可能是三通道的，先把三个通道进行进制转换转为"单通道"的
catch
    I_s = x;
end
    
    
BW = imdilate(I_s,ones(5,5)) ~= imerode(I_s,ones(5,5));%蚕食大小可以修改
BW(I_s==0)=1;%处理细胞隔膜为细胞膜
%真值为细胞膜，假值为非细胞膜
% BW=zeros(size(x));
% BW(imdilate(I_s,ones(5,5)) ~= imerode(I_s,ones(5,5)))=255;
% BW(I_s==0)=255;
%[~, threshold] = edge(I_s, 'sobel');%也是一个获得边界的方法
%BW = edge(I_s, 'sobel', 'nothinning', threshold*0.001);
end