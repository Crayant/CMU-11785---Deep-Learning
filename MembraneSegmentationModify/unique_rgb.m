function out = unique_rgb(in)
%置换数组维度，进制转换后的第三维度就没有用了，所以会被第四维度所置换
out = permute(double(in(:,:,3,:)) +...
    double(in(:,:,2,:))*256 + double(in(:,:,1,:))*256^2,[1 2 4 3]);


