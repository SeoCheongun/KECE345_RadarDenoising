%% prepare_dataset.m
% Extract RAR files into class folders and prepare them for MATLAB training.
% Outputs:
%   1) extracted_data/      : extracted raw data
%   2) prepared_data/       : image copies grouped by label
%   3) dataset_for_training.mat : train/validation/test data

clear; clc;

%% Settings
projectDir = fileparts(mfilename("fullpath"));
archivePattern = "*.rar";

extractedDir = fullfile(projectDir, "extracted_data");
preparedDir = fullfile(projectDir, "prepared_data");
outputMat = fullfile(projectDir, "dataset_for_training.mat");

imageSize = [224 224 3];      % Change to [64 64 3], [128 128 3], etc. if needed.
trainRatio = 0.70;
valRatio = 0.15;
testRatio = 0.15;
customExtractorPath = "";     % Example: "C:\Program Files\7-Zip\7z.exe"

rng(42);                      % Reproducible split

%% 1. Find archives
archives = dir(fullfile(projectDir, archivePattern));
if isempty(archives)
    error("No %s files found in: %s", archivePattern, projectDir);
end

if ~exist(extractedDir, "dir")
    mkdir(extractedDir);
end
if ~exist(preparedDir, "dir")
    mkdir(preparedDir);
end

%% 2. Extract RAR files
extractor = findExtractor(customExtractorPath);
fprintf("Found %d archive files.\n", numel(archives));

for i = 1:numel(archives)
    archivePath = fullfile(archives(i).folder, archives(i).name);
    [~, className] = fileparts(archives(i).name);
    classExtractDir = fullfile(extractedDir, className);

    if exist(classExtractDir, "dir") && ~isempty(dir(fullfile(classExtractDir, "**", "*.*")))
        fprintf("[%s] already extracted. Skipping.\n", className);
        continue;
    end

    if extractor.kind == "none"
        fprintf("No RAR extractor found. Skipping automatic extraction.\n");
        break;
    end

    if ~exist(classExtractDir, "dir")
        mkdir(classExtractDir);
    end

    fprintf("[%s] extracting...\n", archives(i).name);
    extractArchive(archivePath, classExtractDir, extractor);
end

%% 3. Build file index
allFiles = listFiles(extractedDir);
if isempty(allFiles)
    error(["No extracted files found.\n" ...
           "Install 7-Zip/WinRAR, or set customExtractorPath near the top of this script,\n" ...
           "or manually extract each RAR into extracted_data/<class_name>/ and run again.\n" ...
           "Example folder: extracted_data/B/, extracted_data/FTW/"]);
end

labels = labelsFromTopFolder(allFiles, extractedDir);
fileTable = table(string(allFiles(:)), categorical(labels(:)), ...
    'VariableNames', ["File", "Label"]);

writetable(fileTable, fullfile(projectDir, "file_index.csv"));
fprintf("Saved file index: file_index.csv (%d files)\n", height(fileTable));

%% 4. If the data contains images, build image datastores
imageExts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"];
[~, ~, exts] = cellfun(@fileparts, allFiles, 'UniformOutput', false);
isImage = ismember(lower(string(exts)), imageExts);

if any(isImage)
    imageFiles = string(allFiles(isImage));
    imageLabels = labelsFromTopFolder(cellstr(imageFiles), extractedDir);

    makePreparedImageFolders(imageFiles, imageLabels, preparedDir);

    imds = imageDatastore(preparedDir, ...
        "IncludeSubfolders", true, ...
        "LabelSource", "foldernames", ...
        "ReadFcn", @(x)readAndFixImage(x, imageSize));

    [imdsTrain, imdsRest] = splitEachLabel(imds, trainRatio, "randomized");
    restValRatio = valRatio / (valRatio + testRatio);
    [imdsValidation, imdsTest] = splitEachLabel(imdsRest, restValRatio, "randomized");

    adsTrain = augmentedImageDatastore(imageSize(1:2), imdsTrain);
    adsValidation = augmentedImageDatastore(imageSize(1:2), imdsValidation);
    adsTest = augmentedImageDatastore(imageSize(1:2), imdsTest);

    save(outputMat, ...
        "imds", "imdsTrain", "imdsValidation", "imdsTest", ...
        "adsTrain", "adsValidation", "adsTest", ...
        "fileTable", "imageSize", "-v7.3");

    fprintf("\nImage training dataset created.\n");
    fprintf("Train: %d, Validation: %d, Test: %d\n", ...
        numel(imdsTrain.Files), numel(imdsValidation.Files), numel(imdsTest.Files));
    fprintf("Saved file: %s\n", outputMat);
    return;
end

%% 5. Otherwise, try to convert numeric files into a feature matrix
fprintf("No image files found. Trying to build a numeric feature matrix.\n");

[X, Y, usedFiles] = buildNumericFeatureMatrix(fileTable);
if isempty(X)
    error("Could not find image or numeric data files. Check file_index.csv.");
end

cv1 = cvpartition(Y, "HoldOut", 1 - trainRatio);
idxTrain = training(cv1);
idxRest = test(cv1);

YRest = Y(idxRest);
cv2 = cvpartition(YRest, "HoldOut", testRatio / (valRatio + testRatio));
restIndices = find(idxRest);
idxValidation = false(size(Y));
idxTest = false(size(Y));
idxValidation(restIndices(training(cv2))) = true;
idxTest(restIndices(test(cv2))) = true;

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain);
XValidation = X(idxValidation, :);
YValidation = Y(idxValidation);
XTest = X(idxTest, :);
YTest = Y(idxTest);

save(outputMat, ...
    "X", "Y", "XTrain", "YTrain", "XValidation", "YValidation", "XTest", "YTest", ...
    "usedFiles", "fileTable", "-v7.3");

fprintf("\nNumeric training dataset created.\n");
fprintf("X size: %d x %d\n", size(X, 1), size(X, 2));
fprintf("Train: %d, Validation: %d, Test: %d\n", ...
    numel(YTrain), numel(YValidation), numel(YTest));
fprintf("Saved file: %s\n", outputMat);

%% Local functions
function extractor = findExtractor(customExtractorPath)
    if strlength(customExtractorPath) > 0 && isfile(customExtractorPath)
        [~, name] = fileparts(customExtractorPath);
        if contains(lower(name), "winrar")
            kind = "winrar";
        elseif contains(lower(name), "unrar")
            kind = "unrar";
        else
            kind = "7z";
        end
        extractor = struct("kind", kind, "path", customExtractorPath);
        return;
    end

    candidates = [
        "7z", "7z";
        "7z", "C:\Program Files\7-Zip\7z.exe";
        "7z", "C:\Program Files\NVIDIA Corporation\NVIDIA GeForce Experience\7z.exe";
        "winrar", "C:\Program Files\WinRAR\WinRAR.exe";
        "winrar", "C:\Program Files (x86)\WinRAR\WinRAR.exe";
        "unrar", "unrar"
    ];

    extractor = struct("kind", "none", "path", "");
    for k = 1:size(candidates, 1)
        kind = candidates(k, 1);
        path = candidates(k, 2);

        if contains(path, filesep) || contains(path, ":")
            if isfile(path)
                extractor.kind = kind;
                extractor.path = path;
                return;
            end
        else
            [status, out] = system("where " + path);
            if status == 0
                lines = splitlines(strtrim(out));
                extractor.kind = kind;
                extractor.path = string(lines(1));
                return;
            end
        end
    end
end

function extractArchive(archivePath, outDir, extractor)
    switch extractor.kind
        case "7z"
            cmd = sprintf('"%s" x -y -o"%s" "%s"', extractor.path, outDir, archivePath);
        case "winrar"
            cmd = sprintf('"%s" x -ibck -y "%s" "%s\\"', extractor.path, archivePath, outDir);
        case "unrar"
            cmd = sprintf('"%s" x -y "%s" "%s\\"', extractor.path, archivePath, outDir);
        otherwise
            error("Unsupported extractor.");
    end

    [status, out] = system(cmd);
    if status ~= 0
        error("Extraction failed:\n%s\n%s", archivePath, out);
    end
end

function files = listFiles(rootDir)
    d = dir(fullfile(rootDir, "**", "*"));
    d = d(~[d.isdir]);
    files = fullfile({d.folder}, {d.name});
    files = files(:);
end

function labels = labelsFromTopFolder(files, rootDir)
    labels = strings(numel(files), 1);
    rootDir = char(rootDir);

    for k = 1:numel(files)
        rel = erase(string(files{k}), string(rootDir) + filesep);
        parts = split(rel, filesep);
        labels(k) = parts(1);
    end
end

function makePreparedImageFolders(imageFiles, imageLabels, preparedDir)
    if ~exist(preparedDir, "dir")
        mkdir(preparedDir);
    end

    for k = 1:numel(imageFiles)
        label = string(imageLabels(k));
        labelDir = fullfile(preparedDir, label);
        if ~exist(labelDir, "dir")
            mkdir(labelDir);
        end

        [~, name, ext] = fileparts(imageFiles(k));
        dst = fullfile(labelDir, name + "_" + k + ext);
        if ~isfile(dst)
            copyfile(imageFiles(k), dst);
        end
    end
end

function img = readAndFixImage(filename, imageSize)
    img = imread(filename);

    if size(img, 3) == 1 && imageSize(3) == 3
        img = repmat(img, 1, 1, 3);
    elseif size(img, 3) == 4
        img = img(:, :, 1:3);
    end

    img = imresize(img, imageSize(1:2));
end

function [X, Y, usedFiles] = buildNumericFeatureMatrix(fileTable)
    vectors = {};
    labels = strings(0, 1);
    usedFiles = strings(0, 1);

    for k = 1:height(fileTable)
        file = fileTable.File(k);
        label = string(fileTable.Label(k));

        try
            v = readNumericVector(file);
        catch
            continue;
        end

        if isempty(v) || any(~isfinite(v))
            continue;
        end

        vectors{end + 1, 1} = v(:)'; %#ok<AGROW>
        labels(end + 1, 1) = label; %#ok<AGROW>
        usedFiles(end + 1, 1) = file; %#ok<AGROW>
    end

    if isempty(vectors)
        X = [];
        Y = categorical.empty(0, 1);
        return;
    end

    maxLen = max(cellfun(@numel, vectors));
    X = zeros(numel(vectors), maxLen);
    for k = 1:numel(vectors)
        v = vectors{k};
        X(k, 1:numel(v)) = v;
    end

    Y = categorical(labels);
    X = normalize(X, 2);
end

function v = readNumericVector(file)
    [~, ~, ext] = fileparts(file);
    ext = lower(ext);

    switch ext
        case {".csv", ".txt", ".dat", ".tsv"}
            opts = detectImportOptions(file);
            tbl = readtable(file, opts);
            numericVars = varfun(@isnumeric, tbl, "OutputFormat", "uniform");
            data = table2array(tbl(:, numericVars));
            v = data(:);
        case ".mat"
            s = load(file);
            names = fieldnames(s);
            v = [];
            for i = 1:numel(names)
                value = s.(names{i});
                if isnumeric(value)
                    v = [v; numericToFeatureVector(value)]; %#ok<AGROW>
                end
            end
        case ".wav"
            [audio, fs] = audioread(file);
            audio = mean(audio, 2);
            coeff = mfcc(audio, fs);
            v = coeff(:);
        otherwise
            v = [];
    end
end

function v = numericToFeatureVector(x)
    x = double(x);

    if isvector(x)
        x = x(:);
    end

    % Small real arrays can be used directly. Large time-domain signals are
    % summarized so the final X matrix stays trainable on a normal PC.
    if isreal(x) && numel(x) <= 20000
        v = x(:);
        return;
    end

    if isvector(x)
        x = x(:).';
    end

    mag = abs(x);
    xr = real(x);
    xi = imag(x);

    magMean = mean(mag, 2);
    magStd = std(mag, 0, 2);
    magMax = max(mag, [], 2);
    magRms = sqrt(mean(mag.^2, 2));
    realMean = mean(xr, 2);
    realStd = std(xr, 0, 2);
    imagMean = mean(xi, 2);
    imagStd = std(xi, 0, 2);

    v = [magMean; magStd; magMax; magRms; realMean; realStd; imagMean; imagStd];
end
