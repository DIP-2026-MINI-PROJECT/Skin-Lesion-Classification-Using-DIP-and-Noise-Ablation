function [label, proba] = predict_image(imgPath, bundle, saveFig)
%PREDICT_IMAGE  Classify one image and show every DIP pipeline stage.
%   [label, proba] = predict_image('data/suspicious/suspicious_0001.png')
%   [label, proba] = predict_image(path, bundle, 'out.png')   % also save figure
if nargin < 2 || isempty(bundle)
    f = fullfile(fileparts(mfilename('fullpath')), '..', 'models', 'lesion_classifier.mat');
    bundle = getfield(load(f, 'bundle'), 'bundle');
end
if nargin < 3, saveFig = ''; end

img = imread(imgPath);
[feats, stages] = extract_features(img);
z = (feats - bundle.mu) ./ bundle.sigma;
[pred, score] = predict(bundle.model, z);
label = char(string(pred));
classes = bundle.model.ClassNames;
proba = score(1, :);

fprintf('Prediction: %s  (this is a screening aid, not a diagnosis)\n', label);
for c = 1:numel(proba)
    fprintf('  %-12s %.1f%%\n', string(classes(c)), 100*proba(c));
end

order = {'original','hair_mask','hair_removed','enhanced','spectrum', ...
         'denoised','sharpened','raw_mask','lesion_mask','lesion_only'};
fig = figure('Position', [100 100 1200 900]);
for i = 1:numel(order)
    subplot(3, 4, i);
    imshow(stages.(order{i}));
    title(order{i}, 'Interpreter', 'none');
end
t = sprintf('Prediction: %s (screening aid, not a diagnosis)', label);
for c = 1:numel(proba)
    t = sprintf('%s  %s=%.1f%%', t, string(classes(c)), 100*proba(c));
end
sgtitle(t);
if ~isempty(saveFig)
    print(fig, saveFig, '-dpng', '-r130');
    fprintf('Saved stage visualization -> %s\n', saveFig);
end
end
