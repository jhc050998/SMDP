function labels = readMNISTLabels(filename)

fid=fopen(filename, 'r'); %导入文件

%读取前8字节
magic = fread(fid, 1, 'int32', 0, 'ieee-be');
numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');

%读取标签
labels = fread(fid, inf, 'unsigned char');

fclose(fid);
end

