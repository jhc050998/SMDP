function [derivh] = bp_hidden(Ni, Nh, scaleRate, partialo2, partialo3, partialo4, subderivh) 
derivh = zeros(Ni, Nh); % 与wh同型
part1 = derivh;    
part2 = part1;
for h = 1: Nh  % 循环每个隐层神经元
    if isempty(subderivh{h}) % 隐含层无脉冲发出，则直接跳过
        continue;
    end
%     size(part1(:, h))
%     size(subderivh{h})
%     size(partialo4{h}(:,end))
%     fprintf("ttt")
    part1(:, h) = subderivh{h} * partialo4{h}(:,end); % dV(t3)/dt2 * dt2/dw12
    n = length(partialo3{1}); % 对应输出层t3的个数
    if n > 1 % 决定有无part2
        subderiv = - subderivh{h} * partialo4{h} ./ partialo3{1};
        part2(:, h) = subderiv(:, 1:n-1) * partialo2{1}(:, end);  
    end  
    derivh(:, h) = part1(:, h) + scaleRate .* part2(:, h); 
end