function images = readMNISTImages(filename)

fid=fopen(filename, 'r'); %�����ļ�

%��ȡǰ16���ֽڣ�һ���ֽ�8λ��
magic = fread(fid, 1, 'int32', 0, 'ieee-be');
numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
numCols = fread(fid, 1, 'int32', 0, 'ieee-be');

%��ȡ��СΪ28*28��ͼƬ
images = fread(fid, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]);

fclose(fid);

images = reshape(images, size(images,1)*size(images,2), size(images,3));
% images = double(images);
end

