function [inpainted, hairMask] = remove_hair(img)
%REMOVE_HAIR  U4: blackhat morphology finds thin dark structures (hair),
%   then those pixels are filled back in from their surroundings.
%   img is RGB uint8. hairMask is logical.
gray = rgb2gray(img);
se = strel('disk', 4, 0);              % 9x9 disk, same footprint as cv2 9x9 ellipse
blackhat = imbothat(gray, se);
hairMask = blackhat > 10;

inpainted = img;
for c = 1:3
    % regionfill = Laplace interpolation (cv2 used Telea inpainting)
    inpainted(:,:,c) = regionfill(img(:,:,c), hairMask);
end
end
