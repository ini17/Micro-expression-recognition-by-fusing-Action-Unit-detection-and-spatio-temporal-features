% Generates all the results for the SIGGRAPH paper at:
% http://people.csail.mit.edu/mrub/vidmag
%
% Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
% Quanta Research Cambridge, Inc.
%
% Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih
% License: Please refer to the LICENCE file
% Date: June 2012
%

clear;

dataDir = './EVM_Matlab/data';
resultsDir = './EVM_Matlab/ResultsSIGGRAPH2012';
casme_dir = 'B:\0_0NewLife\datasets\CASME_2\CASME2-coding-20190701.xlsx';
mmew_dir = 'B:\0_0NewLife\datasets\MMEW_Final\MMEW_Micro_Exp.csv';
smic_dir = 'B:\0_0NewLife\datasets\SMIC\HS_cropped.csv';

mkdir(resultsDir);

% 目前来看这个效果最好，还是得使用空间上拉普拉斯金字塔+时间上巴特沃斯滤波器
% inFile = fullfile(dataDir,'EP03_02.avi');

%% 测试模块

%% 根据标签读取图像进行放大 
% 读取CASME2标注并根据标注信息读取图像，由此输入EVM进行放大
% 
amp_list = [4, 8, 12, 16];
[NUM, TXT, RAW]=xlsread(smic_dir, 1);
for amp_idx=1.2:0.2:3.0
    for i = 2:165
%         emotion = RAW{i, 7};
        sub = RAW{i,1};
        name = RAW{i,2};
        onset = RAW{i, 3};
    
        % 注意，按理说这里要进行视频插值，但先放一放，之后考虑
    
    
        % 首先使用EVM进行运动放大，使用Cropped部分，样本帧从onset到apex部分
        % 这里删去EVM的保存视频功能，并返回一个n帧的张量
        % inFile = fullfile(dataDir,'EP02_01f/');
%         inFile = ['B:\0_0NewLife\datasets\CASME_2\RAW_selected\sub',RAW{i,1},'\',RAW{i,2},'\'];
        inFile = ['B:\0_0NewLife\0_Papers\SMC\SMIC\Interpolation\Inter_offset_10\sub', num2str(sub),'\', name,'\'];
%         outDir = ['B:/0_0NewLife/CASME_2/Cropped_EVM_', amp_list(amp_idx), '/sub', RAW{i, 1}, '/', RAW{i, 2}, '/'];
%         outDir = append(['B:/0_0NewLife/0_Papers/FGRMER/CASME2/' ...
%             'RAW_selected_EVM/sub'], RAW{i, 1}, '/', RAW{i, 2}, '/', ...
        outDir = append(['B:/0_0NewLife/0_Papers/SMC/SMIC/' ...
            'matlab_EVM/sub'], num2str(RAW{i, 1}), '/', RAW{i, 2}, '/', ...
            num2str(amp_idx, "%.1f"), '/');
        if ~exist(outDir, "dir")
            mkdir(outDir);
        end
        my_butter(inFile, outDir, amp_idx, 48, 0.5, 10, 30, 0, 0);
        % 这里要注意的是，EVM方法需要对前一帧的金字塔进行对比
        % 因此输入N+1帧时，仅得到N帧EVM增强后的输出
        % 由此这里会少一帧输出，注意之后累积光流时，最后一帧的index为offset-1
    end
end


%% 提取光流模块
% 然后将n帧的张量进行灰度处理，得到VolData
% 这一部分见'\工作内容交接\特征提取\累积光流特征\提取光流\comput_of_CASME2.m'

