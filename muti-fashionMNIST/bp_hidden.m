function [derivh_w, derivh_d] = bp_hidden(Ni, Nh, scaleRate, partialo2, partialo3, partialo4, subderivh_w, subderivh_d)
% partialo2：dV(t3)/dtsj  
% partialo3：dV（tsj3）/d（tsj3） 
% partialo4: dV(t3)/dt2  
% subderivh_w: dt2/dw12  
derivh_w = newMST2(Ni, Nh, partialo2, partialo3, partialo4, subderivh_w, scaleRate);
if ~isempty(subderivh_d)  % 有延迟时，计算
    derivh_d = newMST2(Ni, Nh, partialo2, partialo3, partialo4, subderivh_d, scaleRate);
else
    derivh_d = zeros(Ni, Nh);
end
end

function [derivh, subderiv] = newMST2(Ni, Nh, partialo2, partialo3, partialo4, subderivh, scaleRate)
derivh = zeros(Ni, Nh); % 一共有Ni x Nh个改变量
part1 = derivh;    
part2 = part1;
for h = 1: Nh  % 循环每个隐藏层sjy
    if isempty(subderivh{h}) % 如果这个隐藏层没有t2脉冲发出，则直接跳过，梯度=0
        continue;
    end
    % part1: dt2/dw12  * dV(t3)/dt2 = dV(t3)/dw12 
    part1(:, h) = subderivh{h} * partialo4{h}(:,end); 
    n = length(partialo3{1}); % 对应输出层t3的个数
    if n > 1  % 决定有无part2
        subderiv = - subderivh{h} * partialo4{h} ./ partialo3{1};  

        %  part2：  * dV(t*)/dt3  dt2/dw12 /- (dV（tsj3）/d（tsj3）) * dV(t3)/dt2 
        part2(:, h) = subderiv(:, 1:n-1) * partialo2{1}(:, end);  
        % dt2/dw12 *  dV(t3)/dt2 /- (dV（tsj3）/d（tsj3）) *dV(t3)/dtsj  
    end  
    derivh(:, h) = part1(:, h) + scaleRate .* part2(:, h);  % 隐藏层梯度改变量  直接梯度，间接梯度
end

end