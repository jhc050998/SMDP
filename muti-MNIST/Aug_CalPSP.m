function [psp_m, psp_s, psp] = Aug_CalPSP(ti)
global tau_m tau_s Vnorm Timeline
psp_m = zeros(size(ti,1), length(Timeline)); % (N,40)         
psp_s = psp_m;
for k = 1: size(ti,2)
    temp = Timeline - ti(:,k); % (N,40) 
    temp(temp <= 0) = Inf;
    psp_m = psp_m + exp(-temp/tau_m); 
    psp_s = psp_s + exp(-temp/tau_s);
end
psp = Vnorm .* (psp_m - psp_s); % (N,40)