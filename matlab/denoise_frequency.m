function out = denoise_frequency(img, cutoffRatio)
%DENOISE_FREQUENCY  U2: DFT low-pass filtering of the L channel.
%   Keeps a circle of low frequencies (smooth structure), discards the
%   rest (noise), and inverse-transforms back to the spatial domain.
if nargin < 2, cutoffRatio = 0.15; end

lab = rgb2lab(img);
L = lab(:,:,1);
[rows, cols] = size(L);

F = fftshift(fft2(L));

crow = floor(rows/2) + 1;              % location of DC after fftshift
ccol = floor(cols/2) + 1;
radius = floor(cutoffRatio * min(rows, cols));
[X, Y] = meshgrid(1:cols, 1:rows);
lowpass = ((X - ccol).^2 + (Y - crow).^2) <= radius^2;

Lf = real(ifft2(ifftshift(F .* lowpass)));
lab(:,:,1) = min(max(Lf, 0), 100);
out = lab2rgb(lab, 'OutputType', 'uint8');
end
