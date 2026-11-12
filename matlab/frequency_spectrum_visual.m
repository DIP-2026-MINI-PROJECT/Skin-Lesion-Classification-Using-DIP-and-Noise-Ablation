function spec = frequency_spectrum_visual(img)
%FREQUENCY_SPECTRUM_VISUAL  U2: viewable log-magnitude DFT spectrum (uint8).
gray = double(rgb2gray(img));
F = fftshift(fft2(gray));
mag = log(abs(F) + 1);
spec = uint8(255 * mat2gray(mag));
end
