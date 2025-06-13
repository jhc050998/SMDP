function images = readMNISTImages(filename)

fid=fopen(filename, 'r'); %导入文件

%读取前16个字节（一个字节8位）
magic = fread(fid, 1, 'int32', 0, 'ieee-be');
numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
numCols = fread(fid, 1, 'int32', 0, 'ieee-be');

%读取大小为28*28的图片
images = fread(fid, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]);

fclose(fid);

images = reshape(images, size(images,1)*size(images,2), size(images,3));
% images = double(images);
end

