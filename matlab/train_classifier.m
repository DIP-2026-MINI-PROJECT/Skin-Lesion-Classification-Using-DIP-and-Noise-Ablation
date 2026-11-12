function bundle = train_classifier(dataDir)
%TRAIN_CLASSIFIER  Extract features from every image, train an RBF-SVM and a
%   Random Forest, keep the better one and save it to ../models/.
if nargin < 1
    dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
end
rng(42);

imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
labels  = imds.Labels;
classes = categories(labels);
fprintf('Classes found: %s\n', strjoin(classes', ', '));

n = numel(imds.Files);
nf = numel(feature_names());
X = zeros(n, nf);
tic
for i = 1:n
    X(i,:) = extract_features(readimage(imds, i));
    if mod(i, 20) == 0, fprintf('  features %d/%d\n', i, n); end
end
fprintf('Loaded %d samples, %d features each, in %.1fs\n', n, nf, toc);

cv  = cvpartition(labels, 'HoldOut', 0.25);     % stratified by class
Xtr = X(training(cv),:);  ytr = labels(training(cv));
Xte = X(test(cv),:);      yte = labels(test(cv));

mu = mean(Xtr);  sigma = std(Xtr);  sigma(sigma == 0) = 1;
Xtr = (Xtr - mu) ./ sigma;
Xte = (Xte - mu) ./ sigma;

svm = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'BoxConstraint', 10, ...
              'KernelScale', 'auto', 'ClassNames', classes);
svm = fitPosterior(svm);                         % so predict() returns probabilities

rf = fitcensemble(Xtr, ytr, 'Method', 'Bag', 'NumLearningCycles', 300, ...
                  'Learners', templateTree('NumVariablesToSample', round(sqrt(nf))));

cands = struct('name', {'svm_rbf', 'random_forest'}, 'model', {svm, rf});
bestAcc = -1;
for k = 1:numel(cands)
    pred = categorical(cellstr(string(predict(cands(k).model, Xte))), classes);;
    cm   = confusionmat(yte, pred);
    acc  = sum(diag(cm)) / sum(cm(:));
    prec = diag(cm) ./ max(sum(cm,1)', 1);
    rec  = diag(cm) ./ max(sum(cm,2),  1);
    fprintf('\n=== %s ===  accuracy=%.3f\n', cands(k).name, acc);
    for c = 1:numel(classes)
        fprintf('  %-12s precision=%.3f recall=%.3f\n', classes{c}, prec(c), rec(c));
    end
    disp('Confusion matrix:'); disp(cm);
    if acc > bestAcc
        bestAcc = acc; best = cands(k);
    end
end
fprintf('\nBest model: %s (accuracy=%.3f)\n', best.name, bestAcc);

bundle = struct('model', compact(best.model), 'mu', mu, 'sigma', sigma, ...
                'classes', {classes}, 'featureNames', {feature_names()}, ...
                'modelName', best.name);

% --- Added for the noise-ablation experiment (see run_noise_ablation.m) ------
% Remember exactly which images were held out for testing, so the ablation can
% re-evaluate this SAME fixed model on the SAME test images. These three lines
% only append fields to the bundle; the split, training and evaluation above
% are untouched, and predict_image.m ignores the extra fields.
bundle.testFiles    = cellstr(imds.Files(test(cv)));   % held-out image paths
bundle.testLabels   = cellstr(yte);                    % their true labels
bundle.testAccuracy = bestAcc;                         % accuracy printed above
% ------------------------------------------------------------------------------

outDir = fullfile(fileparts(mfilename('fullpath')), '..', 'models');
if ~exist(outDir, 'dir'), mkdir(outDir); end
save(fullfile(outDir, 'lesion_classifier.mat'), 'bundle');
fprintf('Saved model -> %s\n', fullfile(outDir, 'lesion_classifier.mat'));

if strcmp(best.name, 'random_forest')
    imp = predictorImportance(best.model);
    [~, ord] = sort(imp, 'descend');
    fprintf('\nTop 10 most useful features:\n');
    for j = ord(1:min(10, end))
        fprintf('  %-25s %.4f\n', bundle.featureNames{j}, imp(j));
    end
end
end
