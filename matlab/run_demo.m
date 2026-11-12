% RUN_DEMO  Train on ../data, then classify one sample and save the figure.
here = fileparts(mfilename('fullpath'));
addpath(here);

bundle = train_classifier();

outDir = fullfile(here, '..', 'outputs');
if ~exist(outDir, 'dir'), mkdir(outDir); end
predict_image(fullfile(here, '..', 'data', 'suspicious', 'suspicious_0001.png'), ...
              bundle, fullfile(outDir, 'matlab_demo.png'));
