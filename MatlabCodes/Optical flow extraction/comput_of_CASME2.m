[NUM, TXT, RAW]=xlsread('CASME2-coding-20140508',1);
stride = 4;
for i = 2:255
    sub = RAW{i,1};
    name = RAW{i,2};
    fname = ['B:\0_0NewLife\CASME_2\Cropped\sub',RAW{i,1},'\',RAW{i,2},'\'];
    % vdata = ReadVoldata_mask_CASME2(fname);
    vdata = ReadVoldata_smic(fname);
    % 这里的vdata返回了对应标签对应的全部Cropped灰度图像数据
    of = computOpticalFlow(vdata, stride);
    save(['B:\JetBrains\PyCharmProjects\' ...
        ['0_Postgraduate\MER Practice\MyFirstPaperCode\' ...
        'Cumulative_OpticalFlow\\sub'],sub,'_', name], 'of');
end





