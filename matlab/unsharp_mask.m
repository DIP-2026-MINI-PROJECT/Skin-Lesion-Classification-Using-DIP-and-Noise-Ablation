function out = unsharp_mask(img, sigma, k)
%UNSHARP_MASK  U3: unsharp masking (spatial sharpening), applied per RGB channel.
%   mask = img - lowpass(img);   out = img + k * mask
%   The low-pass filter is a Gaussian (sigma = 2 px, imgaussfilt default 9x9 kernel, on the
%   256x256 images); k = 1 is plain unsharp masking (no high-boost, k > 1).
%   img is RGB uint8; out is RGB uint8.
if nargin < 2, sigma = 2; end
if nargin < 3, k = 1; end

f    = double(img);
blur = imgaussfilt(f, sigma);                   % Gaussian low-pass, per channel
out  = uint8(min(max(round(f + k * (f - blur)), 0), 255));
end
