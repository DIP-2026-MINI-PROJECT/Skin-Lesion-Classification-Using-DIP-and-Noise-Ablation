function results = run_noise_ablation(bundle)
%RUN_NOISE_ABLATION  Noise-robustness ablation: Gaussian vs salt-and-pepper noise.
%
%   results = run_noise_ablation()          % uses ../models/lesion_classifier.mat
%   results = run_noise_ablation(bundle)    % uses a bundle returned by train_classifier()
%
%   EXPERIMENT
%     clean training data -> train ONCE -> evaluate that SAME fixed model on
%       1) the clean held-out test images,
%       2) the same test images + Gaussian noise          (3 severities),
%       3) the same test images + salt-and-pepper noise   (3 severities).
%     Nothing is retrained per noise condition, and the original image files
%     are never modified (noise is added to in-memory copies only).
%
%   The test images / labels are exactly the 25 % hold-out split made by
%   train_classifier.m (it now records them in the saved bundle). Every
%   condition uses the same images, in the same order, with the same labels.
%
%   Noise is injected into the RAW input image, before the project's normal
%   pipeline (extract_features -> run_pipeline: hair removal, CLAHE, DFT
%   low-pass, unsharp masking, segmentation, features). So the result measures
%   the robustness of the WHOLE system, including its own denoising stages.
%
%   OUTPUTS (all in ../outputs/noise_ablation/): results tables (.csv/.md),
%   confusion matrices, metric-vs-severity plots, example noisy images,
%   degradation table and a plain-text analysis report. See the printed list.

here    = fileparts(mfilename('fullpath'));
addpath(here);
projDir = fullfile(here, '..');

%% =========================================================================
%  USER-EDITABLE SETTINGS  -  change noise severities / seed here
%% =========================================================================
cfg.seed           = 42;                    % master seed for ALL noise generation
cfg.gaussianSigma  = [5 15 25];             % Gaussian noise std, in 0-255 grey levels
cfg.saltPepperFrac = [0.01 0.05 0.10];      % fraction of PIXELS corrupted (0.05 = 5 %)
cfg.severityNames  = {'Low', 'Medium', 'High'};  % one name per level above (same length!)
cfg.positiveClass  = 'suspicious';          % class treated as "positive" for P / R / F1 / AUC
cfg.sigDropPct     = 5;                     % "significant" = relative F1 drop >= this many %
cfg.exampleTestIdx = 1;                     % which test image to show in the example figure
cfg.showFigures    = false;                 % true = leave figure windows open
cfg.dataDir        = fullfile(projDir, 'data');                      % only used to re-locate images
cfg.outDir         = fullfile(projDir, 'outputs', 'noise_ablation'); % where everything is saved
%% =========================================================================

nSev = numel(cfg.severityNames);
if numel(cfg.gaussianSigma) ~= nSev || numel(cfg.saltPepperFrac) ~= nSev
    error('run_noise_ablation:config', ...
        'cfg.gaussianSigma, cfg.saltPepperFrac and cfg.severityNames must have the same length.');
end
if ~exist(cfg.outDir, 'dir'), mkdir(cfg.outDir); end

%% ---- 1. Get the trained model + the held-out test split -----------------
if nargin < 1, bundle = []; end
if isempty(bundle)
    modelFile = fullfile(projDir, 'models', 'lesion_classifier.mat');
    if exist(modelFile, 'file') == 2
        S = load(modelFile, 'bundle');
        bundle = S.bundle;
    end
end
if isempty(bundle) || ~isfield(bundle, 'testFiles')
    fprintf(['\n[noise ablation] No saved model with a recorded test split was found.\n' ...
             '                 Running train_classifier() once (unchanged procedure, rng(42)) ...\n\n']);
    bundle = train_classifier();
end

classes      = cellstr(bundle.classes);
modelClasses = cellstr(bundle.model.ClassNames);
if numel(classes) ~= 2
    error('run_noise_ablation:classes', 'This ablation assumes a binary problem (found %d classes).', numel(classes));
end
posRow = find(strcmp(classes,      cfg.positiveClass), 1);   % row/col in confusion matrix
posCol = find(strcmp(modelClasses, cfg.positiveClass), 1);   % column of predict() scores
if isempty(posRow) || isempty(posCol)
    error('run_noise_ablation:positiveClass', ...
        'cfg.positiveClass = ''%s'' is not one of: %s', cfg.positiveClass, strjoin(classes', ', '));
end

testFiles  = bundle.testFiles(:);
testLabels = bundle.testLabels(:);
nTest      = numel(testFiles);

% Read the original test images ONCE. They are never written back to disk;
% every noisy version below is a fresh in-memory copy.
imgs = cell(nTest, 1);
for i = 1:nTest
    imgs{i} = imread(resolve_test_file(testFiles{i}, testLabels{i}, cfg.dataDir));
end
fprintf('\nNoise ablation: model = %s, %d test images (%s), positive class = %s\n', ...
    bundle.modelName, nTest, strjoin(cellfun(@(c) sprintf('%d %s', sum(strcmp(testLabels, c)), c), ...
    classes', 'UniformOutput', false), ', '), cfg.positiveClass);

%% ---- 2. Define the conditions: Clean + Gaussian x levels + S&P x levels --
conds = build_conditions(cfg);
nCond = numel(conds);
gRows = find(strcmp({conds.noise}, 'Gaussian'));
sRows = find(strcmp({conds.noise}, 'Salt-Pepper'));
metricNames = {'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'};
nMet = numel(metricNames);

%% ---- 3. Evaluate the SAME fixed model on every condition ----------------
M       = zeros(nCond, nMet);             % headline metrics (positive class)
CM      = zeros(2, 2, nCond);             % confusion matrices (rows = true, cols = predicted)
preds   = cell(nTest, nCond);             % predicted label of every image / condition
scores  = zeros(nTest, nCond);            % P(positive class) of every image / condition
mets    = cell(nCond, 1);                 % full metric structs (per-class, macro, ...)
tAll = tic;
for ci = 1:nCond
    fprintf('\n[%d/%d] %s\n', ci, nCond, cond_label(conds(ci)));
    Z = zeros(nTest, numel(bundle.mu));
    for i = 1:nTest
        % Same image i, same label i in every condition; only the noise differs.
        noisy = apply_condition(imgs{i}, conds(ci), cfg, i);
        try
            f = extract_features(noisy);                  % unchanged project pipeline
        catch ME
            error('run_noise_ablation:featureFailure', ...
                'extract_features failed for test image %d (%s) under "%s": %s', ...
                i, testFiles{i}, cond_label(conds(ci)), ME.message);
        end
        Z(i, :) = (f - bundle.mu) ./ bundle.sigma;        % same standardisation as training
        if mod(i, 20) == 0, fprintf('  features %d/%d\n', i, nTest); end
    end

    [pred, score] = predict(bundle.model, Z);             % same call as predict_image.m
    pred = cellstr(pred);  pred = pred(:);

    m = compute_metrics(testLabels, pred, score(:, posCol), classes, posRow);
    mets{ci}       = m;
    M(ci, :)       = [m.acc, m.prec, m.rec, m.f1, m.auc];
    CM(:, :, ci)   = m.cm;
    preds(:, ci)   = pred;
    scores(:, ci)  = score(:, posCol);
    fprintf('  acc=%.3f  precision=%.3f  recall=%.3f  F1=%.3f  AUC=%.3f\n', M(ci, :));
end
fprintf('\nAll %d conditions evaluated in %.1f s.\n', nCond, toc(tAll));

% Sanity check: the clean condition must reproduce the accuracy that
% train_classifier.m printed for this model (same images, same pipeline).
if isfield(bundle, 'testAccuracy') && abs(M(1, 1) - bundle.testAccuracy) > 1e-9
    warning('run_noise_ablation:cleanMismatch', ...
        ['Clean accuracy here (%.4f) differs from the test accuracy recorded at training (%.4f). ' ...
         'The saved model and the current pipeline may be out of sync - re-run train_classifier().'], ...
        M(1, 1), bundle.testAccuracy);
elseif isfield(bundle, 'testAccuracy')
    fprintf('Check passed: clean accuracy %.4f matches train_classifier.m (%.4f).\n', M(1, 1), bundle.testAccuracy);
end

%% ---- 4. Degradation relative to the clean baseline ----------------------
%   metric_drop = clean_metric - noisy_metric      (positive = got worse)
%   pct_drop    = 100 * metric_drop / clean_metric
D   = M(1, :) - M;                                  % row 1 (clean) is 0 by definition
Pct = 100 * D ./ M(1, :);
Pct(:, M(1, :) == 0) = NaN;                         % undefined if the clean metric is 0

% Paired exact McNemar test on correctness (accuracy) of noisy vs clean.
correct = false(nTest, nCond);
for ci = 1:nCond, correct(:, ci) = strcmp(preds(:, ci), testLabels); end
mcP = NaN(nCond, 1);
for ci = 2:nCond, mcP(ci) = mcnemar_exact_p(correct(:, 1), correct(:, ci)); end

%% ---- 5. Tables (console + csv + markdown) --------------------------------
print_results_table(M, conds, metricNames);
save_tables(cfg, conds, M, D, Pct, CM, mets, mcP, classes, posRow, nTest, ...
            testFiles, testLabels, preds, scores, metricNames);

%% ---- 6. Plots -------------------------------------------------------------
files = save_plots(cfg, conds, M, CM, mets, gRows, sRows, metricNames, imgs, testFiles, testLabels, classes);

%% ---- 7. Written degradation analysis (derived only from the numbers) ------
report = build_analysis(cfg, conds, M, D, Pct, mcP, gRows, sRows, metricNames, nTest, classes, scores);
fid = fopen(fullfile(cfg.outDir, 'analysis_report.txt'), 'w');
fprintf(fid, 'Noise ablation analysis  (generated %s)\n\n', datestr(now));
for k = 1:numel(report), fprintf(fid, '%s\n', report{k}); end
fclose(fid);
fprintf('\n');
for k = 1:numel(report), fprintf('%s\n', report{k}); end

%% ---- 8. Return / save everything -----------------------------------------
results = struct('cfg', cfg, 'conditions', conds, 'metricNames', {metricNames}, ...
    'metrics', M, 'drop', D, 'pctDrop', Pct, 'confusion', CM, 'classes', {classes}, ...
    'mcnemarP', mcP, 'predictions', {preds}, 'scores', scores, ...
    'testFiles', {testFiles}, 'testLabels', {testLabels}, 'analysis', {report}, 'plots', files);
save(fullfile(cfg.outDir, 'noise_ablation_results.mat'), 'results');

fprintf('\nResults saved in: %s\n', cfg.outDir);
fprintf('  results_table.csv / .md        main table (Condition, Severity, Accuracy, Precision, Recall, F1, AUC)\n');
fprintf('  results_detailed.csv           + per-class / macro metrics, TP/FP/FN/TN, noise parameters\n');
fprintf('  degradation_table.csv          drop and %% drop vs clean, McNemar p-values\n');
fprintf('  confusion_matrices.csv / .png  confusion matrix of every condition\n');
fprintf('  metrics_vs_severity.png        F1 / Accuracy / Precision / Recall vs severity (+ one PNG per metric)\n');
fprintf('  noise_examples.png             one image under all conditions\n');
fprintf('  per_image_predictions.csv      prediction + score of every image in every condition\n');
fprintf('  analysis_report.txt            written conclusions;  noise_ablation_results.mat = all numbers\n');
end

% =========================================================================
%  Local helper functions
% =========================================================================

function conds = build_conditions(cfg)
%BUILD_CONDITIONS  Clean, then Gaussian low/med/high, then salt-pepper low/med/high.
conds = struct('noise', {}, 'severity', {}, 'param', {}, 'level', {});
conds(end+1) = struct('noise', 'Clean', 'severity', 'None', 'param', NaN, 'level', 0);
for k = 1:numel(cfg.severityNames)
    conds(end+1) = struct('noise', 'Gaussian', 'severity', cfg.severityNames{k}, ...
                          'param', cfg.gaussianSigma(k), 'level', k);
end
for k = 1:numel(cfg.severityNames)
    conds(end+1) = struct('noise', 'Salt-Pepper', 'severity', cfg.severityNames{k}, ...
                          'param', cfg.saltPepperFrac(k), 'level', k);
end
end

function s = cond_label(c)
%COND_LABEL  Human-readable description of one condition (ASCII only).
switch c.noise
    case 'Clean',       s = 'Clean (no noise)';
    case 'Gaussian',    s = sprintf('Gaussian, %s (sigma = %g on 0-255 scale)', c.severity, c.param);
    case 'Salt-Pepper', s = sprintf('Salt-and-pepper, %s (%g%% of pixels)', c.severity, 100 * c.param);
end
end

function noisy = apply_condition(img, c, cfg, testIdx)
%APPLY_CONDITION  Return a noisy COPY of img for condition c (img is untouched).
%   The seed depends only on (master seed, noise type, test-image index), so the
%   noise for a given image is identical no matter the loop order, and the three
%   severities of one noise type share a noise pattern (nested severities).
switch c.noise
    case 'Clean'
        noisy = img;
    case 'Gaussian'
        noisy = add_gaussian_noise(img, c.param, cfg.seed + 100000 + testIdx);
    case 'Salt-Pepper'
        noisy = add_salt_pepper_noise(img, c.param, cfg.seed + 200000 + testIdx);
end
end

function f = resolve_test_file(f, label, dataDir)
%RESOLVE_TEST_FILE  Use the recorded path; if the project was moved, look in dataDir/<label>/.
if exist(f, 'file') == 2, return; end
[~, name, ext] = fileparts(f);
alt = fullfile(dataDir, label, [name ext]);
if exist(alt, 'file') == 2
    f = alt;
else
    error('run_noise_ablation:missingImage', 'Test image not found: %s (also tried %s)', f, alt);
end
end

function m = compute_metrics(yTrue, yPred, posScore, classes, posRow)
%COMPUTE_METRICS  Same formulas as train_classifier.m, plus F1 and AUC.
%   cm rows = true class, cols = predicted class, in the order of `classes`.
%   prec/rec/f1 (scalars) are for the POSITIVE class; *Class are per class; macro* average classes.
nC = numel(classes);
cm = zeros(nC);
for r = 1:nC
    for c = 1:nC
        cm(r, c) = sum(strcmp(yTrue, classes{r}) & strcmp(yPred, classes{c}));
    end
end
if sum(cm(:)) ~= numel(yTrue)
    error('run_noise_ablation:labels', 'Some predicted labels are not in the class list.');
end
precC = diag(cm) ./ max(sum(cm, 1)', 1);          % identical to train_classifier.m
recC  = diag(cm) ./ max(sum(cm, 2),  1);
f1C   = 2 * precC .* recC ./ max(precC + recC, eps);

m.cm = cm;
m.acc = sum(diag(cm)) / sum(cm(:));
m.precClass = precC;  m.recClass = recC;  m.f1Class = f1C;
m.prec = precC(posRow);  m.rec = recC(posRow);  m.f1 = f1C(posRow);
m.macroPrec = mean(precC);  m.macroRec = mean(recC);  m.macroF1 = mean(f1C);
negRow = setdiff(1:nC, posRow);
m.tp = cm(posRow, posRow);  m.fn = cm(posRow, negRow);
m.fp = cm(negRow, posRow);  m.tn = cm(negRow, negRow);
m.auc = auc_from_scores(posScore(:), strcmp(yTrue, classes{posRow}));
end

function auc = auc_from_scores(scores, isPos)
%AUC_FROM_SCORES  ROC AUC via the rank-sum (Mann-Whitney) identity; ties get average ranks.
nP = sum(isPos);  nN = sum(~isPos);
if nP == 0 || nN == 0, auc = NaN; return, end
[s, order] = sort(scores);
n = numel(s);
rk = zeros(n, 1);
i = 1;
while i <= n
    j = i;
    while j < n && s(j + 1) == s(i), j = j + 1; end
    rk(i:j) = (i + j) / 2;                         % average rank of a tie group
    i = j + 1;
end
r = zeros(n, 1);  r(order) = rk;
auc = (sum(r(isPos)) - nP * (nP + 1) / 2) / (nP * nN);
end

function p = mcnemar_exact_p(cleanOK, noisyOK)
%MCNEMAR_EXACT_P  Two-sided exact McNemar test on paired correct/incorrect outcomes.
%   Only images whose correctness CHANGED matter: b = right->wrong, c = wrong->right.
b = sum(cleanOK & ~noisyOK);
c = sum(~cleanOK & noisyOK);
n = b + c;
if n == 0, p = 1; return, end
k = 0:min(b, c);
p = min(1, 2 * sum(exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) - n * log(2))));
end

function print_results_table(M, conds, metricNames)
fprintf('\n=== RESULTS (P / R / F1 / AUC are for the positive class) ===\n');
fprintf('%-12s %-9s %-9s %-9s %-9s %-9s %-9s\n', 'Condition', 'Severity', metricNames{:});
for ci = 1:numel(conds)
    fprintf('%-12s %-9s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f\n', ...
        conds(ci).noise, conds(ci).severity, M(ci, :));
end
end

% ---------------------------- saving tables ------------------------------

function save_tables(cfg, conds, M, D, Pct, CM, mets, mcP, classes, posRow, nTest, ...
                     testFiles, testLabels, preds, scores, metricNames)
nCond = numel(conds);
out = @(name) fullfile(cfg.outDir, name);

% ---- main table (exactly the requested columns) ----
hdr = [{'Condition', 'Severity'}, metricNames];
rows = cell(nCond, numel(hdr));
for ci = 1:nCond
    rows(ci, :) = [{conds(ci).noise, conds(ci).severity}, num2cell(M(ci, :))];
end
write_csv(out('results_table.csv'), hdr, rows);

% ---- detailed table ----
pc = classes{posRow};
hdr2 = {'Condition', 'Severity', 'Gaussian_sigma_0to255', 'SaltPepper_pixel_pct', 'N_test', ...
        'Accuracy', ['Precision_' pc], ['Recall_' pc], ['F1_' pc], 'AUC', ...
        'Macro_precision', 'Macro_recall', 'Macro_F1', ...
        ['Precision_' classes{1}], ['Recall_' classes{1}], ['F1_' classes{1}], ...
        ['Precision_' classes{2}], ['Recall_' classes{2}], ['F1_' classes{2}], ...
        ['TP_' pc], ['FP_' pc], ['FN_' pc], ['TN_' pc]};
rows2 = cell(nCond, numel(hdr2));
for ci = 1:nCond
    m = mets{ci};
    sg = NaN; sp = NaN;
    if strcmp(conds(ci).noise, 'Gaussian'),    sg = conds(ci).param;       end
    if strcmp(conds(ci).noise, 'Salt-Pepper'), sp = 100 * conds(ci).param; end
    rows2(ci, :) = {conds(ci).noise, conds(ci).severity, sg, sp, nTest, ...
        m.acc, m.prec, m.rec, m.f1, m.auc, m.macroPrec, m.macroRec, m.macroF1, ...
        m.precClass(1), m.recClass(1), m.f1Class(1), m.precClass(2), m.recClass(2), m.f1Class(2), ...
        m.tp, m.fp, m.fn, m.tn};
end
write_csv(out('results_detailed.csv'), hdr2, rows2);

% ---- degradation table (noisy conditions only) ----
hdr3 = [{'Condition', 'Severity'}, strcat(metricNames, '_drop'), strcat(metricNames, '_pct_drop'), ...
        {'McNemar_p_accuracy_vs_clean'}];
rows3 = cell(nCond - 1, numel(hdr3));
for ci = 2:nCond
    rows3(ci - 1, :) = [{conds(ci).noise, conds(ci).severity}, num2cell(D(ci, :)), ...
                        num2cell(Pct(ci, :)), {mcP(ci)}];
end
write_csv(out('degradation_table.csv'), hdr3, rows3);

% ---- confusion matrices (long format: one row per condition x true class) ----
hdr4 = [{'Condition', 'Severity', 'True_class'}, strcat('Pred_', classes(:)')];
rows4 = cell(nCond * numel(classes), numel(hdr4));
r = 0;
for ci = 1:nCond
    for t = 1:numel(classes)
        r = r + 1;
        rows4(r, :) = [{conds(ci).noise, conds(ci).severity, classes{t}}, num2cell(CM(t, :, ci))];
    end
end
write_csv(out('confusion_matrices.csv'), hdr4, rows4);

% ---- per-image predictions (proves every condition used the same images/labels) ----
hdr5 = {'Test_index', 'File', 'True_label'};
for ci = 1:nCond
    key = [conds(ci).noise '_' conds(ci).severity];
    hdr5 = [hdr5, {[key '_pred'], [key '_P_' pc]}]; %#ok<AGROW>
end
rows5 = cell(numel(testFiles), numel(hdr5));
for i = 1:numel(testFiles)
    [~, nm, ex] = fileparts(testFiles{i});
    row = {i, [nm ex], testLabels{i}};
    for ci = 1:nCond, row = [row, {preds{i, ci}, scores(i, ci)}]; end %#ok<AGROW>
    rows5(i, :) = row;
end
write_csv(out('per_image_predictions.csv'), hdr5, rows5);

% ---- markdown (paste-able into a report) ----
fid = fopen(out('results_table.md'), 'w');
fprintf(fid, '## Noise ablation results (same fixed model, %d test images)\n\n', nTest);
fprintf(fid, 'Precision, Recall, F1 and AUC are for the positive class "%s".\n\n', pc);
md_table(fid, hdr, rows);
fprintf(fid, '\n### Drop vs clean (clean - noisy; positive = worse)\n\n');
r6 = cell(nCond - 1, 2 + numel(metricNames));
for ci = 2:nCond, r6(ci - 1, :) = [{conds(ci).noise, conds(ci).severity}, num2cell(D(ci, :))]; end
md_table(fid, hdr(1:2 + numel(metricNames)), r6);
fprintf(fid, '\n### Percentage drop vs clean (100 * drop / clean)\n\n');
for ci = 2:nCond, r6(ci - 1, :) = [{conds(ci).noise, conds(ci).severity}, num2cell(Pct(ci, :))]; end
md_table(fid, hdr(1:2 + numel(metricNames)), r6);
fclose(fid);
end

function write_csv(file, header, rows)
%WRITE_CSV  Tiny dependency-free CSV writer. NaN -> empty field.
fid = fopen(file, 'w');
if fid < 0, error('run_noise_ablation:io', 'Cannot write %s', file); end
fprintf(fid, '%s\n', strjoin(header, ','));
for r = 1:size(rows, 1)
    parts = cell(1, size(rows, 2));
    for c = 1:size(rows, 2)
        parts{c} = csv_field(rows{r, c});
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
end

function s = csv_field(v)
if ischar(v)
    s = v;
    if any(s == ',' | s == '"'), s = ['"' strrep(s, '"', '""') '"']; end
elseif isnan(v)
    s = '';
elseif v == fix(v)
    s = sprintf('%d', v);
else
    s = sprintf('%.6f', v);
end
end

function md_table(fid, header, rows)
fprintf(fid, '| %s |\n', strjoin(header, ' | '));
fprintf(fid, '|%s\n', repmat('---|', 1, numel(header)));
for r = 1:size(rows, 1)
    parts = cell(1, size(rows, 2));
    for c = 1:size(rows, 2)
        v = rows{r, c};
        if ischar(v), parts{c} = v;
        elseif isnan(v), parts{c} = '';
        else, parts{c} = sprintf('%.4f', v);
        end
    end
    fprintf(fid, '| %s |\n', strjoin(parts, ' | '));
end
end

% ------------------------------- plots ------------------------------------

function files = save_plots(cfg, conds, M, CM, mets, gRows, sRows, metricNames, imgs, testFiles, testLabels, classes)
nSev = numel(cfg.severityNames);
colG = [0.00 0.45 0.74];   colS = [0.85 0.33 0.10];

% x tick labels: one line each (a newline inside a tick label makes MATLAB split it into
% separate ticks). The severity values are spelled out in the x-axis label instead.
xt = [{'Clean'}, cfg.severityNames(:)'];
sg = strjoin(arrayfun(@(v) sprintf('%g', v),       cfg.gaussianSigma,  'UniformOutput', false), '/');
sp = strjoin(arrayfun(@(v) sprintf('%g', 100 * v), cfg.saltPepperFrac, 'UniformOutput', false), '/');
sevNote = sprintf('Noise severity  (%s:  Gaussian sigma = %s,  salt & pepper = %s %% of pixels)', ...
                  strjoin(cfg.severityNames, '/'), sg, sp);

% shared y-range for all metric plots so panels are directly comparable
sel = [4 1 2 3];                                    % F1, Accuracy, Precision, Recall
allv = M(:, sel);
ylo = max(0, floor((min(allv(:)) - 0.05) * 20) / 20);
yl  = [ylo, 1.05];

% ---- 2x2 combined figure ----
fig = new_figure(cfg, [80 80 1100 800]);
for q = 1:4
    subplot(2, 2, q);
    draw_metric(M, gRows, sRows, sel(q), metricNames{sel(q)}, xt, yl, colG, colS, sevNote);
end
files.metricsCombined = fullfile(cfg.outDir, 'metrics_vs_severity.png');
print(fig, files.metricsCombined, '-dpng', '-r130');
if ~cfg.showFigures, close(fig); end

% ---- one PNG per metric ----
for q = 1:4
    fig = new_figure(cfg, [80 80 620 460]);
    draw_metric(M, gRows, sRows, sel(q), metricNames{sel(q)}, xt, yl, colG, colS, sevNote);
    fn = fullfile(cfg.outDir, sprintf('%s_vs_severity.png', lower(metricNames{sel(q)})));
    print(fig, fn, '-dpng', '-r130');
    files.(['metric_' lower(metricNames{sel(q)})]) = fn;
    if ~cfg.showFigures, close(fig); end
end

% ---- confusion matrices (all conditions, one colour scale) ----
nCond = numel(conds);
fig = new_figure(cfg, [60 60 1250 650]);
cmax = max(sum(CM(:, :, 1), 2));                    % largest class size -> common colour scale
blues = [linspace(1, 0.05, 64)', linspace(1, 0.30, 64)', linspace(1, 0.60, 64)'];
ncol = ceil((nCond + 1) / 2);                       % 2 rows; last tile is a legend/notes tile
for ci = 1:nCond
    ax = subplot(2, ncol, ci);
    cmI = CM(:, :, ci);
    imagesc(cmI);  colormap(blues);  set(ax, 'CLim', [0 cmax]);
    axis(ax, 'image');
    set(ax, 'XTick', 1:2, 'XTickLabel', classes, 'YTick', 1:2, 'YTickLabel', classes, 'FontSize', 8, ...
        'XColor', 'k', 'YColor', 'k');
    xlabel('Predicted', 'Color', 'k');  ylabel('True', 'Color', 'k');
    for r = 1:2
        for c = 1:2
            tc = 'k';  if cmI(r, c) > 0.55 * cmax, tc = 'w'; end
            text(c, r, sprintf('%d', cmI(r, c)), 'HorizontalAlignment', 'center', ...
                 'FontSize', 13, 'FontWeight', 'bold', 'Color', tc);
        end
    end
    title(sprintf('%s %s  (acc %.3f)', conds(ci).noise, conds(ci).severity, mets{ci}.acc), ...
          'FontSize', 9, 'Interpreter', 'none', 'Color', 'k');
end
ax = subplot(2, ncol, nCond + 1);  axis(ax, 'off');
text(0, 1, {'Confusion matrices', '', 'rows = true class', 'columns = predicted class', ...
            sprintf('positive class = %s', cfg.positiveClass), '', 'same fixed model,', 'same test images'}, ...
     'VerticalAlignment', 'top', 'FontSize', 10, 'Interpreter', 'none', 'Color', 'k');
files.confusion = fullfile(cfg.outDir, 'confusion_matrices.png');
print(fig, files.confusion, '-dpng', '-r130');
if ~cfg.showFigures, close(fig); end

% ---- example images: ONE test image under every condition ----
ei = min(max(round(cfg.exampleTestIdx), 1), numel(imgs));
fig = new_figure(cfg, [60 60 1200 640]);
ecol = nSev + 1;                                    % columns: Clean/notes + one per severity
for ci = 1:nCond
    % Same apply_condition() + seeds as the evaluation -> exactly what the model saw.
    im = apply_condition(imgs{ei}, conds(ci), cfg, ei);
    if ci == 1,                    pos = 1;                             % top-left: clean
    elseif ci <= 1 + nSev,         pos = ci;                            % Gaussian: top row
    else,                          pos = ecol + 1 + (ci - 1 - nSev);    % salt-pepper: bottom row
    end
    subplot(2, ecol, pos);
    image(im);  axis image off;
    if ci == 1, ttl = 'Clean';
    elseif strcmp(conds(ci).noise, 'Gaussian')
        ttl = sprintf('Gaussian %s (sigma=%g)', conds(ci).severity, conds(ci).param);
    else
        ttl = sprintf('Salt & pepper %s (%g%%)', conds(ci).severity, 100 * conds(ci).param);
    end
    title(ttl, 'FontSize', 10, 'Interpreter', 'none', 'Color', 'k');
end
ax = subplot(2, ecol, ecol + 1);  axis(ax, 'off');   % bottom-left: notes tile
[~, nm, ex] = fileparts(testFiles{ei});
text(0, 1, {'Same test image,', 'every condition', '', [nm ex], ['label: ' testLabels{ei}], '', ...
            sprintf('seed = %d', cfg.seed), 'Gaussian sigma on 0-255 scale', 'S&P = % of pixels corrupted'}, ...
     'VerticalAlignment', 'top', 'FontSize', 10, 'Interpreter', 'none', 'Color', 'k');
files.examples = fullfile(cfg.outDir, 'noise_examples.png');
print(fig, files.examples, '-dpng', '-r130');
if ~cfg.showFigures, close(fig); end
end

function fig = new_figure(cfg, pos)
vis = 'off';  if cfg.showFigures, vis = 'on'; end
fig = figure('Visible', vis, 'Position', pos, 'Color', 'w');
end

function draw_metric(M, gRows, sRows, mi, mname, xt, yl, colG, colS, sevNote)
%DRAW_METRIC  One metric vs severity; Gaussian and salt-and-pepper on the same axes.
%   Both curves start at the same clean point (x = 0).
x  = 0:numel(gRows);
yG = [M(1, mi); M(gRows, mi)];
yS = [M(1, mi); M(sRows, mi)];
plot(x, yG, '-o', 'Color', colG, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', colG);
hold on;
plot(x, yS, '-s', 'Color', colS, 'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', colS);
hold off;
grid on;
set(gca, 'XTick', x, 'XTickLabel', xt, 'XLim', [-0.25, x(end) + 0.25], 'YLim', yl, 'FontSize', 9, ...
    'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.3 0.3 0.3]);
xlabel(sevNote, 'FontSize', 8, 'Color', 'k');
ylabel(mname, 'Color', 'k');
title(sprintf('%s vs noise severity', mname), 'FontSize', 11, 'Color', 'k');
lgd = legend({'Gaussian', 'Salt & pepper'}, 'Location', 'southwest');
set(lgd, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.5 0.5 0.5]);
end

% --------------------------- degradation analysis -------------------------

function L = build_analysis(cfg, conds, M, D, Pct, mcP, gRows, sRows, metricNames, nTest, classes, scores)
%BUILD_ANALYSIS  Turn the numbers into plain-text conclusions. Nothing here is
%   hard-coded: every statement is computed from M / D / Pct / McNemar p-values.
tol = 1e-9;
ACC = 1;  F1 = 4;
nSev = numel(gRows);
sev  = cfg.severityNames;
typeRows  = {gRows, sRows};
typeNames = {'Gaussian', 'Salt-and-pepper'};
cmp = @(a, b) double(a - b > tol) - double(b - a > tol);   % +1 a larger, -1 b larger, 0 equal

L = {};
L{end+1} = '=== DEGRADATION ANALYSIS (computed from the results above; nothing assumed) ===';
L{end+1} = sprintf(['Definitions: drop = clean - noisy (positive = worse, negative = improved); %% drop = 100*drop/clean.\n' ...
    'Precision/Recall/F1/AUC are for the positive class "%s". "Significant" (practical) = relative F1 drop >= %g%%.\n' ...
    'Statistical check = exact McNemar test on per-image correctness vs clean (p < 0.05 marked *).\n' ...
    'Test set: %d images, so one image = %.2f accuracy points. Noise is one fixed-seed realisation per image.'], ...
    cfg.positiveClass, cfg.sigDropPct, nTest, 100 / nTest);
L{end+1} = '';

% ---- 1. Which noise type degrades more? (mean F1 drop over the severities) ----
dG = D(gRows, F1);  dS = D(sRows, F1);
L{end+1} = '1) Which noise type causes greater degradation?  (criterion: mean F1 drop over the severities)';
if max([dG; dS]) <= tol
    L{end+1} = '   Neither: no noisy condition lowered F1 below the clean baseline.';
else
    c = cmp(mean(dG), mean(dS));
    if c == 0
        L{end+1} = sprintf('   Both degrade F1 equally on average (mean drop %.4f).', mean(dG));
    else
        if c > 0, w = 'Gaussian'; else, w = 'Salt-and-pepper'; end
        L{end+1} = sprintf('   %s noise causes the greater degradation.', w);
    end
    L{end+1} = sprintf('   Mean F1 drop:  Gaussian %.4f (%.1f%% of clean)   Salt-and-pepper %.4f (%.1f%% of clean)', ...
        mean(dG), nmean(Pct(gRows, F1)), mean(dS), nmean(Pct(sRows, F1)));
    for k = 1:nSev
        c = cmp(dG(k), dS(k));
        lbl = {'equal', 'Gaussian larger', 'Salt-and-pepper larger'};
        switch c, case 0, who = lbl{1}; case 1, who = lbl{2}; otherwise, who = lbl{3}; end
        L{end+1} = sprintf('   %-7s F1 drop: Gaussian %.4f vs Salt-and-pepper %.4f  -> %s', ...
            [sev{k} ':'], dG(k), dS(k), who);
    end
    ca = cmp(mean(D(gRows, ACC)), mean(D(sRows, ACC)));
    cf = cmp(mean(dG), mean(dS));
    if ca == cf
        L{end+1} = sprintf('   Accuracy gives the same ranking (mean drop: Gaussian %.4f, Salt-and-pepper %.4f).', ...
            mean(D(gRows, ACC)), mean(D(sRows, ACC)));
    else
        L{end+1} = sprintf('   NOTE: accuracy ranks the two noise types differently (mean drop: Gaussian %.4f, Salt-and-pepper %.4f).', ...
            mean(D(gRows, ACC)), mean(D(sRows, ACC)));
    end
end
L{end+1} = '';

% ---- 2. At which severity does degradation become significant? ----
L{end+1} = sprintf('2) At which severity does degradation become significant?  (practical: F1 drop >= %g%%)', cfg.sigDropPct);
for t = 1:2
    rows = typeRows{t};
    k = find(Pct(rows, F1) >= cfg.sigDropPct, 1);
    [mx, kk] = max(Pct(rows, F1));
    if isempty(k)
        L{end+1} = sprintf('   %-16s threshold never reached (largest F1 drop %s at %s).', typeNames{t}, pct_str(mx), sev{kk});
    else
        r = rows(k);
        L{end+1} = sprintf('   %-16s first reached at %s: F1 %.4f -> %.4f (drop %.1f%%).', typeNames{t}, sev{k}, M(1, F1), M(r, F1), Pct(r, F1));
    end
    s = '';
    for k = 1:nSev
        star = '';  if mcP(rows(k)) < 0.05, star = '*'; end
        s = [s, sprintf('%s p=%.3f%s  ', sev{k}, mcP(rows(k)), star)]; %#ok<AGROW>
    end
    L{end+1} = sprintf('   %-16s McNemar (accuracy vs clean): %s', '', strtrim(s));
end
L{end+1} = '';

% ---- 3. Which metric is affected the most? ----
meanPct = zeros(1, numel(metricNames));
for j = 1:numel(metricNames), meanPct(j) = nmean(Pct(2:end, j)); end
L{end+1} = '3) Which metric is affected the most?  (criterion: mean % drop over all noisy conditions)';
L{end+1} = ['   ' strjoin(cellfun(@(n, v) sprintf('%s %s', n, pct_str(v)), metricNames, num2cell(meanPct), 'UniformOutput', false), ',  ')];
[mx, ~] = max(meanPct);
if isnan(mx) || mx <= tol
    L{end+1} = '   No metric dropped on average.';
else
    top = find(abs(meanPct - mx) <= tol);
    L{end+1} = sprintf('   Most affected overall: %s (mean drop %.1f%%).', strjoin(metricNames(top), ' / '), mx);
end
for t = 1:2
    mp = zeros(1, numel(metricNames));
    for j = 1:numel(metricNames), mp(j) = nmean(Pct(typeRows{t}, j)); end
    [mxt, ~] = max(mp);
    if isnan(mxt) || mxt <= tol
        L{end+1} = sprintf('   Under %s noise: no metric dropped on average.', typeNames{t});
    else
        top = find(abs(mp - mxt) <= tol);
        L{end+1} = sprintf('   Under %s noise: %s (mean drop %.1f%%).', typeNames{t}, strjoin(metricNames(top), ' / '), mxt);
    end
end
[mxd, ix] = max(reshape(D(2:end, :), [], 1));
if mxd > tol
    [ri, cj] = ind2sub(size(D(2:end, :)), ix);
    L{end+1} = sprintf('   Largest single drop: %s %s, %s: %.4f -> %.4f (-%.4f).', ...
        conds(ri + 1).noise, conds(ri + 1).severity, metricNames{cj}, M(1, cj), M(ri + 1, cj), mxd);
end
L{end+1} = '';

% ---- 4. Which noise type declines the steepest? ----
L{end+1} = '4) Which noise type produces the steepest decline?  (criterion: least-squares slope of F1 over Clean,Low,Medium,High)';
slope = zeros(1, 2);
for t = 1:2
    pf = polyfit(0:nSev, [M(1, F1); M(typeRows{t}, F1)]', 1);
    slope(t) = pf(1);
    stepDrop = -diff([M(1, F1); M(typeRows{t}, F1)])' + 0;      % drop at each severity step (+0 avoids "-0")
    [mxs, ks] = max(stepDrop);
    if ks == 1, from = 'Clean'; else, from = sev{ks - 1}; end
    L{end+1} = sprintf('   %-16s slope %+.4f F1 per severity step; largest single step: %s -> %s (drop %.4f).', ...
        typeNames{t}, slope(t), from, sev{ks}, mxs);
end
if max(-slope) <= tol
    L{end+1} = '   Neither noise type shows a declining F1 trend.';
else
    c = cmp(-slope(1), -slope(2));
    if c == 0
        L{end+1} = '   Both noise types decline at the same rate.';
    else
        if c > 0, w = typeNames{1}; else, w = typeNames{2}; end
        L{end+1} = sprintf('   Steepest decline: %s noise.', w);
    end
end
L{end+1} = '';
% AUC is only informative if the model's scores are graded. With (near-)binary scores it equals
% balanced accuracy, so it cannot be read as independent evidence.
nDistinct = zeros(1, size(scores, 2));
for k = 1:size(scores, 2), nDistinct(k) = numel(unique(round(scores(:, k) * 1e6))); end
if max(nDistinct) <= 3
    L{end+1} = sprintf(['Note on AUC: the model outputs only %d-%d distinct scores per condition (saturated votes), so AUC reduces to ' ...
        'balanced accuracy (= accuracy on a balanced test set) and adds no information beyond accuracy.'], min(nDistinct), max(nDistinct));
    L{end+1} = '';
end
L{end+1} = sprintf(['Caution: with %d test images, differences of a few images are within sampling noise; ' ...
    'treat conclusions as indicative, and check the McNemar p-values before calling a difference real.'], nTest);
end

function s = pct_str(v)
%PCT_STR  Format a percentage; 'n/a' when undefined (clean metric was 0).
if isnan(v), s = 'n/a (clean metric is 0)'; else, s = sprintf('%.1f%%', v); end
end

function v = nmean(x)
%NMEAN  Mean ignoring NaNs (NaN if nothing left).
x = x(~isnan(x));
if isempty(x), v = NaN; else, v = mean(x); end
end
