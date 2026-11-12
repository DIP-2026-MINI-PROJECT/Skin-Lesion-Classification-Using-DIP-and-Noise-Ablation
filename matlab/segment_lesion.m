function mask = segment_lesion(img)
%SEGMENT_LESION  U1: Otsu threshold on the Lab 'a' channel (green-red axis).
lab = rgb2lab(img);
a8 = uint8(min(max(round(lab(:,:,2) + 128), 0), 255));   % OpenCV-style 8-bit 'a'
level = graythresh(a8);
mask = imbinarize(a8, level);
end
