% PICK_AND_PREDICT  Choose any image from disk and run it through the trained pipeline.
here = fileparts(mfilename('fullpath'));
addpath(here);

startDir = fullfile(here, '..', 'data');
if ~exist(startDir, 'dir'), startDir = pwd; end
[file, folder] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif', 'Images'}, ...
    'Choose a lesion image', startDir);
if isequal(file, 0), disp('No file chosen.'); return, end

img = imread(fullfile(folder, file));
if size(img,3) == 4, img = img(:,:,1:3); end          % drop alpha channel
img = imresize(img, [256 256]);                       % same scale as the training images

tmp = [tempname '.png'];
imwrite(img, tmp);
predict_image(tmp, [], '');
delete(tmp);