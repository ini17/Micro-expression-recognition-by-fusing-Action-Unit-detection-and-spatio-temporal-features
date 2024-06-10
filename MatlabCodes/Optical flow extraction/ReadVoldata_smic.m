function VolData = ReadVoldata_smic(fname)
%fname is name of img folder
%eg: 'D:\Database\CASME2\Cropped\Cropped\sub01\EP02_01f\'
imglist = dir(fname);
[length , z]= size(imglist);
img_hw = imread([fname,imglist(3).name]);
len = length-2;
[h, w, c] = size(img_hw);
VolData = zeros(h,w,len);
% firstimg = 9999999;
% for i = 3:length  
%     num = str2num(cell2mat(regexp(imglist(i).name,'\d', 'match')));
%     if num < firstimg
%         firstimg = num;
%     end
% end
for i = 1:len   
%     if firstimg < 100000
%         imgname = ['reg_image0', int2str(firstimg), '.bmp'];
%     else
%         imgname = ['reg_image', int2str(firstimg), '.bmp']; 
%     firstimg = firstimg + 1;
    img_rgb = imread([fname, imglist(i+2).name]);
    img = rgb2gray(img_rgb);
    VolData(:,:,i) = img;
end
end

