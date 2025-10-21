clc;
clear all;

%% =========================
%  Parameter Configuration
%  =========================
baseFolderPath = 'F:\\....';
baseOutputFolderPath1 = 'F:\\...';
labelFolderPath = 'F:\\...';

patterns = 1:3;                     % Different wavelength patterns (e.g., RGB or multiple illumination conditions)
numFolders = 20;                    % Number of training folders
numImagesPerFolder = 800;           % Number of images in each folder
energy1 = 95;                       % PCA energy threshold for label data
energy2 = 95;                       % PCA energy threshold for speckle data
imageSize = [256, 256];             % Original image size
scaledSize = [128, 128];            % Rescaled image size for processing

numTrainSamples = 5600;             % Number of training samples
rand = load('random.mat');          % Load random index file for data shuffling,
randIdx = rand.randIdx;    

fprintf('\n speckles load...\n');
tic;
% ============================
% Load speckle patterns for multiple wavelengths
% ============================
for patternIdx = patterns
    folderPath = fullfile(baseFolderPath, sprintf('wavelength%d', patternIdx));
    % Read training speckle images
    X_train = read_tiff_images(folderPath, 1, numFolders, numImagesPerFolder);
    % Read test speckle images
    XX_test = read_tiff_images(folderPath, numFolders+1, numFolders+2, numImagesPerFolder);

    % Dynamically store training/testing data for each wavelength
    eval(sprintf('X_train%d = X_train;', patternIdx));
    eval(sprintf('XX_test%d = XX_test;', patternIdx));
end
toc

%% ============================
% Load corresponding label images
% ============================
fprintf('\n label load...\n');
tic;
for folderIdx = 1:numFolders
    folderName = fullfile(labelFolderPath, num2str(folderIdx));
    files = dir(fullfile(folderName, '*.bmp'));
    for fileIdx = 1:numImagesPerFolder
        img = imread(fullfile(folderName, files(fileIdx).name));
        Y_train(:, :, (folderIdx - 1) * numImagesPerFolder + fileIdx) = img;
    end
end
toc;

% ============================
% Data preparation and reshaping
% ============================
y_train_all = [];
x_train_all = [];
x_test_all = [];

for patternIdx = patterns
    X_train = eval(sprintf('X_train%d;', patternIdx));
    XX_test = eval(sprintf('XX_test%d;', patternIdx));

    % Shuffle training data according to random index
    X_train_shuffled = X_train(:, :, randIdx);

    % Split test set and append external test data
    X_test = double(X_train_shuffled(:, :, numTrainSamples+1:end));
    X_test = double(cat(3, X_test, XX_test));

    % Retain training subset
    X_train_shuffled(:, :, numTrainSamples+1:end) = [];
    X_train_shuffled = double(X_train_shuffled);

    % Flatten image data into column vectors
    x_train_single = reshape(X_train_shuffled, [], size(X_train_shuffled, 3));
    x_test_single = reshape(X_test, [], size(X_test, 3));

    % Concatenate multi-wavelength data
    x_train_all = [x_train_all; x_train_single];
    x_test_all = [x_test_all; x_test_single];
end

% ============================
% Process label data (corresponding images)
% ============================
fprintf('\nProcessing label data...\n');
tic;
Y_train_shuffled = uint8(Y_train(:, :, randIdx));
Y_test = uint8(Y_train_shuffled(:, :, numTrainSamples+1:end));
Y_train_shuffled(:, :, numTrainSamples+1:end) = [];
Y_train_shuffled = double(Y_train_shuffled);

% Flatten label data
y_train = reshape(Y_train_shuffled, [], size(Y_train_shuffled, 3));
clear Y_train Y_train_shuffled;
toc;

% Convert to single precision for GPU processing
y_train=single(y_train);
x_train_all=single(x_train_all);
x_test_all=single(x_test_all);

%% ============================
% GPU-accelerated PCA on label data
% ============================
fprintf('\n  label load (GPU)...\n');
tic;
y_train_gpu = gpuArray(y_train');
toc;

fprintf('\nPerforming PCA on label data (GPU)...\n');
tic;
[coef_y, ~, latent_y] = pca(y_train_gpu);
latent_y = 100 * gather(latent_y) / sum(gather(latent_y));
train_idx_y = find(cumsum(latent_y) > energy1, 1);
M = gather(coef_y(:, 1:train_idx_y));   % Reduced basis for label space
toc;

%% ============================
% GPU-accelerated PCA on speckle data
% ============================
fprintf('\n speckles load (GPU)...\n');
tic;
x_train_gpu = gpuArray(x_train_all');
toc;

fprintf('\nPerforming PCA on speckle data (GPU)...\n');
tic;
[coef_x, ~, latent_x] = pca(x_train_gpu);
latent_x = 100 * gather(latent_x) / sum(gather(latent_x));
train_idx_x = find(cumsum(latent_x) > energy2, 1);
Mx = gather(coef_x(:, 1:train_idx_x));  % Reduced basis for speckle space
toc;

% ============================
% Project training data into PCA subspaces
% ============================
fprintf('\n train speckles project (GPU)...\n');
tic
x_train_pca = Mx' * x_train_gpu';
toc;
clear x_train_gpu coef_x;

fprintf('\n test speckles project (GPU)...\n');
tic;
x_test_pca = Mx' * x_test_all;
toc;

clear X_train X_train1 X_train2 X_train3 XX_test1 XX_test2 XX_test3;
clear X_train_shuffled XX_test X_test x_train_single x_test_single;

%% ============================
% Solve the inverse transmission matrix (ITM)
% ============================
fprintf('\nSolving inverse transmission matrix (GPU)...\n');
tic;
YY = pinv(M) * y_train;                   % Compute pseudoinverse of label PCA basis
toc;

fprintf('\n  ITM_PCA_time   ...\n');
tic;
ITM_PCA = YY * pinv(x_train_pca);         % Estimate inverse transmission matrix
toc;

%% ============================
% Predict and reconstruct test images
% ============================
fprintf('\nPredicting test images (GPU)...\n');
tic;
x_test_pca_gpu = gpuArray(x_test_pca);
test_pred = M * ITM_PCA * gather(x_test_pca_gpu);    % Reconstruction via inverse model
test_pred2 = reshape(test_pred, sqrt(size(test_pred, 1)), sqrt(size(test_pred, 1)), size(test_pred, 2));
test_pred3 = uint8(test_pred2);
clear x_test_pca_gpu;
toc;

%% ============================
% Save reconstructed images
% ============================
clear test_pred test_pred2;
outputFolderPath = fullfile(baseOutputFolderPath1);
numImages = size(test_pred3, 3);

for folderIdx = 1:ceil(numImages / numImagesPerFolder)
    folderName = sprintf('%d', folderIdx);
    folderPath = fullfile(outputFolderPath, folderName);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
    for idx = (folderIdx - 1) * numImagesPerFolder + 1:min(folderIdx * numImagesPerFolder, numImages)
        filename = sprintf('%05d.jpg', idx);
        filepath = fullfile(folderPath, filename);
        img = test_pred3(:, :, idx);
        imwrite(img, filepath);
    end
end

%% ============================
% Function: Read .tiff image stacks from folders
% ============================
function data = read_tiff_images(folderPath, startFolder, endFolder, numImagesPerFolder)
    % This function loads a series of TIFF images from a range of folders,
    % sorts them by numeric order, and stacks them into a 3D matrix.

    data = [];  % Initialize container
    
    for folderIdx = startFolder:endFolder
        folderName = fullfile(folderPath, num2str(folderIdx));
        files = dir(fullfile(folderName, '*.tiff'));

        % Extract numerical indices from filenames and sort naturally
        fileNames = {files.name};
        fileNumbers = regexp(fileNames, '\d+', 'match');
        fileNumbers = cellfun(@(x) str2double(x{end}), fileNumbers);
        [~, sortIdx] = sort(fileNumbers);
        sortedFileNames = fileNames(sortIdx);

        % Load each image and stack into a 3D matrix
        for fileIdx = 1:numImagesPerFolder
            img = imread(fullfile(folderName, sortedFileNames{fileIdx}));
            grayImg = img; % Convert to grayscale if necessary
            data(:, :, (folderIdx - startFolder) * numImagesPerFolder + fileIdx) = grayImg;
        end
    end
end
