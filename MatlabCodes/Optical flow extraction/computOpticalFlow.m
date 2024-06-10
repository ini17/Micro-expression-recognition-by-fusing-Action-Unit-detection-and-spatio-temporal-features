function of = computOpticalFlow(vdata, stride)
    [h w t] = size(vdata);
    frame_num = floor((t-1)/stride);
    % 向下取整
    of = zeros(h,w, 2, frame_num);
    % 这里的第三维度2包括了两个分量，分别是光流的水平分量和垂直分量
    count = 1;
    H=fspecial('gaussian', 5, 5);
    % H为一个预训练过的高斯低通滤波器
    
    for i = 1:frame_num
        % 若t-1不为stride的整数倍，则舍弃后续的部分帧
        im1 = imfilter(vdata(:,:,count), H);
        im2 = imfilter(vdata(:,:,count + stride), H);  
        uv = estimate_flow_ba(im1, im2,'pyramid_levels', 2);
        of(:,:,:,i) = uv; 
        count = count + stride;
    end
end