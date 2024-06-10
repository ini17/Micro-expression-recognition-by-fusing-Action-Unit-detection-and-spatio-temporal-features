[NUM, TXT, RAW]=xlsread('HS_cropped.csv',1);
parallel = 9;
stride = 1;
for amp_factor=1.2:0.2:3.0
    for i = 2:165
        sub = RAW{i,1};
        name = RAW{i,2};
%         fname = ['B:\0_0NewLife\CASME_2\Cropped\sub',RAW{i,1},'\',RAW{i,2},'\'];
        fname = append('B:\0_0NewLife\0_Papers\SMC\SMIC\matlab_EVM\sub', ...
            num2str(sub), '\', name, '\', num2str(amp_factor, '%.1f'), '\');
        fprintf("%s\n", fname);
        % vdata = ReadVoldata_mask_CASME2(fname);
        vdata = ReadVoldata_smic(fname);
        % 这里的vdata返回了对应标签对应的全部Cropped灰度图像数据
        of = computOpticalFlow(vdata, stride);
%         save(['B:\JetBrains\PyCharmProjects\' ... 
%             ['0_Postgraduate\MER Practice\MyFirstPaperCode\' ...
%             'Cumulative_OpticalFlow\\sub'],sub,'_', name], 'of');
        outDir = append(['B:\0_0NewLife\0_Papers\SMC\SMIC\mat\Optical_Flow\', ...
            num2str(amp_factor, '%.1f')]);
        if ~exist(outDir, "dir")
            mkdir(outDir);
        end
        filename = append(outDir, '\sub', num2str(sub), '_', name, ".mat");
        save(filename, 'of');
    end
end





