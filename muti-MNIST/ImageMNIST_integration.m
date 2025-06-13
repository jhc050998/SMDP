function samples = ImageMNIST_integration(pattern, label, num)
[Dim, ~] = size(pattern);
% [label, ind] = sort(label);
% pattern = pattern(ind, :);
K = length(unique(label));
[~, ID] = unique(label);
ID = ID-1;
image_10 = cell(num, K);
temp = cell(num, K);

samples = [];
for i=1:num
    for j=1:K
        temp{i, j} = reshape(pattern(:, ID(j)+i), sqrt(Dim), sqrt(Dim));
        image_10{i, j} = [image_10{i, j}, temp{i, j}];
    end
    samples = [samples; image_10{i,:}];
end

end

