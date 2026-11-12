function noisy = add_gaussian_noise(img, sigma, seed)
%ADD_GAUSSIAN_NOISE  Add zero-mean i.i.d. Gaussian noise to an image.   [noise ablation]
%   noisy = add_gaussian_noise(img, sigma)
%   noisy = add_gaussian_noise(img, sigma, seed)    % reproducible
%
%   sigma is the noise standard deviation on the 0-255 scale (i.e. in uint8
%   grey levels), so sigma = 15 means "+/- 15 grey levels (1 std)".
%
%   Scale handling (so sigma always means the same thing):
%     * uint8 image  (0..255, what this project uses) -> sigma is used as-is.
%     * double/single image in [0,1]                  -> sigma/255 is used.
%   Both are equivalent to  imnoise(img, 'gaussian', 0, (sigma/255)^2).
%
%   Noise is drawn independently for every pixel and colour channel, then the
%   result is clipped back to the valid range (and rounded for uint8).
%
%   seed : optional non-negative integer. The global RNG state is saved before
%          seeding and restored afterwards, so this function never disturbs
%          the caller's random stream (extract_features also reseeds rng).
%
%   The noise FIELD depends only on (seed, image size); sigma just scales it.
%   Using the same seed for Low/Medium/High therefore gives nested severities
%   (High = 5x Low before clipping), which keeps severity curves clean.
%   The input image is never modified.

if nargin < 3, seed = []; end

if ~isempty(seed)
    prevState = rng;                           % save global RNG state
    restoreRng = onCleanup(@() rng(prevState));% ... and restore it on exit
    rng(seed, 'twister');
end

switch class(img)
    case 'uint8'
        maxVal = 255;   sigmaEff = sigma;          % already on the 0-255 scale
    case {'double', 'single'}
        maxVal = 1;     sigmaEff = sigma / 255;    % [0,1] image: convert sigma
    otherwise
        error('add_gaussian_noise:class', 'Unsupported image class: %s', class(img));
end

x = double(img) + sigmaEff * randn(size(img));
x = min(max(x, 0), maxVal);                        % clip to valid range

if isa(img, 'uint8')
    noisy = uint8(round(x));
else
    noisy = cast(x, class(img));
end
end
