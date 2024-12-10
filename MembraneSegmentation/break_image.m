function out =  break_image(im, sq_len)
%打破图片为碎片补丁
s = size(im);
% s=[1024,1024]
% sq_len=128
num_pieces = s(1)/sq_len;
out = zeros(sq_len,sq_len,num_pieces*num_pieces,'uint8');
size(out)
for i = 1:num_pieces
    for j = 1:num_pieces
 
        out(:,:,(i-1)*num_pieces +j) = im( (i-1)*sq_len +1: i*sq_len, (j-1)*sq_len +1: j*sq_len );
        
    end
    
end
end