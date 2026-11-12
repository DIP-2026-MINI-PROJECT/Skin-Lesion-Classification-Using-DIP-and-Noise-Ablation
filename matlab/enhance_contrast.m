function out = enhance_contrast(img)
%ENHANCE_CONTRAST  U3: CLAHE on the L channel of Lab only.
lab = rgb2lab(img);                    % L in [0,100]
L = lab(:,:,1) / 100;
L = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
lab(:,:,1) = L * 100;
out = lab2rgb(lab, 'OutputType', 'uint8');
end
