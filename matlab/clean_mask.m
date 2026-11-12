function out = clean_mask(mask)
%CLEAN_MASK  U4: opening removes specks, closing fills small holes.
se = strel('disk', 3, 0);              % 7x7 disk
out = imclose(imopen(mask, se), se);
end
