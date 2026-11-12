function out = largest_component(mask)
%LARGEST_COMPONENT  U1: connected-component analysis, keep the biggest blob.
cc = bwconncomp(mask, 8);
if cc.NumObjects == 0
    out = mask;
    return
end
[~, idx] = max(cellfun(@numel, cc.PixelIdxList));
out = false(size(mask));
out(cc.PixelIdxList{idx}) = true;
end
