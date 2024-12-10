function out = unique_rgb(in)
%置换数组维度，不知道为何叠加后的维度还是四维
out = permute(double(in(:,:,3,:)) +...
    double(in(:,:,2,:))*256 + double(in(:,:,1,:))*256^2,[1 2 4 3]);


