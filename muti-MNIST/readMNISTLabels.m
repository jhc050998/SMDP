function labels = readMNISTLabels(filename)

fid=fopen(filename, 'r'); %�����ļ�

%��ȡǰ8�ֽ�
magic = fread(fid, 1, 'int32', 0, 'ieee-be');
numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');

%��ȡ��ǩ
labels = fread(fid, inf, 'unsigned char');

fclose(fid);
end

