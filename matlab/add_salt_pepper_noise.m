function noisy = add_salt_pepper_noise(img, fraction, seed)
%ADD_SALT_PEPPER_NOISE  Corrupt a fraction of pixels with black/white.   [noise ablation]
%   noisy = add_salt_pepper_noise(img, fraction)
%   noisy = add_salt_pepper_noise(img, fraction, seed)    % reproducible
%
%   fraction is the share of PIXEL LOCATIONS (height x width) that get
%   corrupted, e.g. 0.05 = 5 % of the pixels. Exactly round(fraction*H*W)
%   pixels are chosen at random. Half become "salt" (white, 255) and half
%   "pepper" (black, 0); all colour channels of a corrupted pixel are set
%   together, so each corrupted pixel is pure black or pure white.
%   (Note: imnoise(img,'salt & pepper',d) instead corrupts individual colour
%   samples, giving coloured specks on RGB images - that is NOT what is done here.)
%
%   seed : optional non-negative integer. The global RNG state is saved before
%          seeding and restored afterwards (see add_gaussian_noise).
%
%   Severities are NESTED for a given seed: the pixels corrupted at 1 % are a
%   subset of those corrupted at 5 %, which are a subset of those at 10 %, and
%   a pixel keeps the same salt/pepper value across severities.
%   The input image is never modified.

if nargin < 3, seed = []; end
if fraction < 0 || fraction > 1
    error('add_salt_pepper_noise:range', 'fraction must be in [0,1].');
end

if ~isempty(seed)
    prevState = rng;                           % save global RNG state
    restoreRng = onCleanup(@() rng(prevState));% ... and restore it on exit
    rng(seed, 'twister');
end

switch class(img)
    case 'uint8',             white = 255;
    case {'double','single'}, white = 1;       % [0,1] image
    otherwise
        error('add_salt_pepper_noise:class', 'Unsupported image class: %s', class(img));
end

[h, w, nch] = size(img);
n  = h * w;
nc = round(fraction * n);                      % number of corrupted pixels

perm = randperm(n);                            % random order of all pixel locations
idx  = perm(1:nc);                             % first nc of them are corrupted
salt   = idx(1:2:end);                         % odd ranks  -> white
pepper = idx(2:2:end);                         % even ranks -> black (exact 50/50 split)

noisy = img;
for c = 1:nch
    ch = noisy(:, :, c);
    ch(salt)   = white;
    ch(pepper) = 0;
    noisy(:, :, c) = ch;
end
end
