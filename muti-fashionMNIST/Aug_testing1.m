function numCorrect = Aug_testing1(data, labels, w1, w2, d1, d2)
global threshold groupSize nargin Nclass
numCorrect = 0;
for i = 1: length(labels) % 循环每个标签
    flag = zeros(Nclass,1);
    if iscell(data)
        ti = data{i};
    else
        ti = data(:, i); 
    end

    if nargin == 0   % 无延迟
        if isempty(w1)      % single layer
            [~, ~, psp] = Aug_CalPSP(ti);
            [outNum, ~, Vo] = Aug_CalOutVo  (threshold, w2, psp); 
        else               
              %% two layer 
            [~, ~, psp_i] = Aug_CalPSP (ti);
            [~, th] = Aug_CalOutVo (threshold, w1, psp_i);
            [~, ~, psp_h] = Aug_CalPSP (th);  
            [outNum, ~,Vo] = Aug_CalOutVo (threshold, w2, psp_h);
        end
    elseif nargin == 1   % 有延迟
        if isempty(w1) && isempty(d1)    % single layer  这 
            [~, ~, psp] = Aug_CalPSP (ti,d2);
            [outNum, ~, Vo] = Aug_CalOutVo  (threshold, w2, psp);       
        else             % two layer
            [~, ~, psp_i] = Aug_CalPSP (ti, d1); 
            [~, th] = Aug_CalOutVo(threshold, w1, psp_i); 
            [~, ~, psp_h] = Aug_CalPSP (th, d2);            
            [outNum, ~,Vo] = Aug_CalOutVo(threshold, w2, psp_h);
        end
    end
    if isempty(Vo)
        continue
    end
    %    解码，优先选输出max的输出神经元，再者选电压最大神经元；
    %   多层输出神经元时，"投票"选出类别
    for group = 1: groupSize   
        Vo_temp = Vo( 1+(group-1)*Nclass: Nclass+(group-1)*Nclass, :); 
        outNum_temp = outNum(1+(group-1)*Nclass: Nclass+(group-1)*Nclass);
        [~, row] = max ( max (Vo_temp,[],2 )); 
        if length(find( outNum_temp == max(outNum_temp) )) == 1  
            flag(outNum_temp == max(outNum_temp)) = flag(outNum_temp == max(outNum_temp)) + 1;  
        elseif all(outNum_temp == 0)  
            flag(row) = flag(row) + 1;
        end  
    end
    if length(find(flag == max(flag))) == 1 &&  find(flag == max(flag)) == labels(i) 
        numCorrect = numCorrect + 1;
    else
        i; 
    end

end

