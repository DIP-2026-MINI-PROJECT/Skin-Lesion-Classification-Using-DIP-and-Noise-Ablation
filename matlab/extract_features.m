function [feats, stages] = extract_features(img)
%EXTRACT_FEATURES  ABCDE-style + GLCM texture + DCT energy feature vector (1x22).
%   (No "E"/Evolution feature: needs photos of the same lesion over time.)
stages = run_pipeline(img);
mask = stages.lesion_mask;
im   = stages.sharpened;

[asymLR, asymTB]       = asymmetry_score(mask);
circ                   = border_irregularity(mask);
[nColors, colorStd]    = color_variation(im, mask, 4);
[area, diam, aspect]   = diameter_features(mask);
glcm                   = glcm_features(im, mask);
dctF                   = dct_energy_features(im, mask, 32);

feats = [asymLR asymTB circ nColors colorStd area diam aspect glcm dctF];
end

% ------------------------------------------------------------------------
function [lr, tb] = asymmetry_score(mask)
[ys, xs] = find(mask);
if isempty(xs), lr = 0; tb = 0; return, end
[h, w] = size(mask);
cx = mean(xs);  cy = mean(ys);
area = max(nnz(mask), 1);

% Reflect the mask about its centroid: flip, then shift so the centroid
% lands back where it started.
flLR = imtranslate(uint8(fliplr(mask)), [round(2*cx - w - 1), 0], 'nearest');
flTB = imtranslate(uint8(flipud(mask)), [0, round(2*cy - h - 1)], 'nearest');
lr = nnz(xor(mask, logical(flLR))) / area;
tb = nnz(xor(mask, logical(flTB))) / area;
end

function c = border_irregularity(mask)
B = bwboundaries(mask, 8, 'noholes');
if isempty(B), c = 0; return, end
[~, i] = max(cellfun(@(b) size(b,1), B));
b = B{i};
if size(b,1) < 3, c = 0; return, end
perim = sum(sqrt(sum(diff(b,1,1).^2, 2)));
area  = polyarea(b(:,2), b(:,1));
if perim == 0, c = 0; return, end
c = min(max(4*pi*area / perim^2, 0), 1);       % circularity
end

function [nSig, colorStd] = color_variation(im, mask, k)
mask3 = repmat(mask, [1 1 3]);
pix = double(reshape(im(mask3), [], 3));
if size(pix,1) < k, nSig = 1; colorStd = 0; return, end
rng(0);                                          % reproducible k-means
[idx, centers] = kmeans(pix, k, 'Replicates', 5, 'Start', 'plus', ...
                        'Options', statset('MaxIter', 100));
counts = accumarray(idx, 1, [k 1]);
nSig = sum(counts > 0.05 * size(pix,1));
colorStd = mean(std(centers, 1, 1));            % population std per channel
end

function [area, diam, aspect] = diameter_features(mask)
[ys, xs] = find(mask);
if isempty(xs), area = 0; diam = 0; aspect = 0; return, end
area = numel(xs);
w = max(xs) - min(xs);
h = max(ys) - min(ys);
diam = max(w, h);
aspect = w / max(h, 1);
end

function f = glcm_features(im, mask)
gray = rgb2gray(im);
gray(~mask) = 0;
q = floor(double(gray) / 32);                    % 8 gray levels, 0..7
offsets = [0 1; -1 1; -1 0];                     % 0, 45, 90 degrees at distance 1
glcm = graycomatrix(q, 'Offset', offsets, 'NumLevels', 8, ...
                    'GrayLimits', [0 7], 'Symmetric', true);
p = graycoprops(glcm, {'Contrast','Homogeneity','Energy','Correlation'});
% MATLAB "Energy" is the angular second moment; sqrt gives the skimage definition
f = [p.Contrast, p.Homogeneity, sqrt(p.Energy), p.Correlation];
f(isnan(f)) = 0;                                 % constant region -> undefined correlation
end

function f = dct_energy_features(im, mask, blockSize)
[ys, xs] = find(mask);
if isempty(xs), f = [0 0]; return, end
gray = rgb2gray(im);
crop = gray(min(ys):max(ys), min(xs):max(xs));
if isempty(crop), f = [0 0]; return, end
r = imresize(crop, [blockSize blockSize]);
E = dct2(double(r)).^2;
k = blockSize / 4;
low = sum(sum(E(1:k, 1:k))) / (sum(E(:)) + 1e-8);
f = [low, 1 - low];
end
