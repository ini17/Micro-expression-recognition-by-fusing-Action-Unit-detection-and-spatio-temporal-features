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

mkdir(resultsDir);


%% baby
inFile = fullfile(dataDir,'baby.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_iir(inFile, resultsDir, 10, 16, 0.4, 0.05, 0.1);

% Alternative processing using butterworth filter
% amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 30, 16, 0.4, 3, 30, 0.1);

%% baby2
inFile = fullfile(dataDir,'baby2.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,150,6, 140/60,160/60,30, 1);

%% camera
inFile = fullfile(dataDir,'camera.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 150, 20, 45, 100, 300, 0);


%% subway
inFile = fullfile(dataDir,'subway.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 60, 90, 3.6, 6.2, 30, 0.3);

%% wrist
%% No mask is used here to generate the output video.
% inFile = fullfile(dataDir,'wrist.mp4');
inFile = fullfile(dataDir,'EP03_02.avi');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_iir(inFile, resultsDir, 10, 16, 0.4, 0.05, 0.1);
% 这个效果挺明显的但是有点振铃

% Alternative processing using butterworth filter
% amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 30, 16, 0.4, 3, 30, 0.1);


%% shadow
% 目前来看这个效果最好，还是得使用空间上拉普拉斯金字塔+时间上巴特沃斯滤波器
% inFile = fullfile(dataDir,'shadow.mp4');
% inFile = fullfile(dataDir,'EP03_02.avi');
inFile = fullfile(dataDir,'EP02_01f/');
fprintf('Processing %s\n', inFile);
my_butter(inFile, resultsDir, 5, 48, 0.5, 10, 30, 0);

%% guitar
% inFile = fullfile(dataDir,'guitar.mp4');
inFile = fullfile(dataDir,'EP03_02.avi');
fprintf('Processing %s\n', inFile);
% 这个会提示内存不足
% amplify E
amplify_spatial_lpyr_temporal_ideal(inFile, resultsDir, 50, 10, 72, 92, 600, 0);
% amplify A
amplify_spatial_lpyr_temporal_ideal(inFile, resultsDir, 100, 10, 100, 120, 600, 0);


%% face
% inFile = fullfile(dataDir,'face.mp4');
inFile = fullfile(dataDir,'EP03_02.avi');
fprintf('Processing %s\n', inFile);
% 这个会出现振铃现象
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,4, ...
                     50/60,60/60,30, 1);


%% face2
% inFile = fullfile(dataDir,'face2.mp4');
inFile = fullfile(dataDir,'EP03_02.avi');
fprintf('Processing %s\n', inFile);

% Motion
amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,80, ...
                                     0.5,10,30, 0);
% Color
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,6, ...
                                     50/60,60/60,30, 1);
