function numCorrect = Aug_testing(data, labels, w1, w2)
global threshold groupSize Nclass
numCorrect = 0;
for i = 1: length(labels) % 循环每个标签
    flag = zeros(Nclass,1);
    if iscell(data)
        ti = data{i};
    else
        ti = data(:, i); 
    end

    [~, ~, psp_i] = Aug_CalPSP(ti);
    [~, ~, th, ~] = Aug_CalOut(threshold, w1, psp_i);
    [~, ~, psp_h] = Aug_CalPSP(th);  
    [outNum, ~,~, Vo] = Aug_CalOut(threshold, w2, psp_h);
    
    if isempty(Vo)
        continue
    end
    
    % 优先选输出max的输出神经元，再者选电压最大神经元；多层输出神经元时，"投票"选出类别
    for group = 1: groupSize   
%         size(Vo)
%         size(outNum)
        Vo_temp = Vo(1+(group-1)*Nclass: Nclass+(group-1)*Nclass, :); 
        outNum_temp = outNum(1+(group-1)*Nclass: Nclass+(group-1)*Nclass);
        [~, row] = max(max(Vo_temp,[],2)); 
        
        if length(find(outNum_temp == max(outNum_temp))) == 1  
            flag(outNum_temp == max(outNum_temp)) = flag(outNum_temp == max(outNum_temp)) + 1;  
        elseif all(outNum_temp == 0)  
            flag(row) = flag(row) + 1;
        end  
    end
    
    % 发射得绝对最多的那一个对应标签，认为正确
    if length(find(flag == max(flag))) == 1 &&  find(flag == max(flag)) == labels(i) 
        numCorrect = numCorrect + 1;
    end
end