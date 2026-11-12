function stages = run_pipeline(img)
%RUN_PIPELINE  Full DIP pipeline. img is an RGB uint8 image.
%   Units used: U1 color models/connectivity, U2 DFT, U3 CLAHE + unsharp masking, U4 morphology.
if size(img,3) == 1, img = repmat(img, [1 1 3]); end
if ~isa(img, 'uint8'), img = im2uint8(img); end

[hairRemoved, hairMask] = remove_hair(img);
enhanced   = enhance_contrast(hairRemoved);
spectrum   = frequency_spectrum_visual(enhanced);
denoised   = denoise_frequency(enhanced);
sharpened  = unsharp_mask(denoised);            % unsharp masking, k = 1

rawMask    = segment_lesion(sharpened);
lesionMask = largest_component(clean_mask(rawMask));
lesionOnly = sharpened .* uint8(lesionMask);      % implicit expansion over RGB

stages = struct('original', img, 'hair_mask', hairMask, ...
    'hair_removed', hairRemoved, 'enhanced', enhanced, ...
    'spectrum', spectrum, 'denoised', denoised, 'sharpened', sharpened, ...
    'raw_mask', rawMask, ...
    'lesion_mask', lesionMask, 'lesion_only', lesionOnly);
end
