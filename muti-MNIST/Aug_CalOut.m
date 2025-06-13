function [outNum, to1, to2, Vo, V0, cj] = Aug_CalOut(theta, w, psp)
% outNum-����Ԫ��������������cj-��ʱ�̸���Ԫ�����������
% to1-���������¼���Ƿ��䣻to2-ͬʱ�̵ķ��������¼��Vo-ģ���Ĥ��λ
global tau_m dt Timeline
factor = exp(-dt/tau_m ); % ʱ��˥������
No = size(w,2); % �����Ԫ����
outNum = zeros(No,1); % ��¼�������Ԫ�����������
to1 = []; to2 = []; cj = [];
V0 = zeros(No,length(Timeline));
Vo = zeros(No,length(Timeline)); % (N,40)���������Ӧ��ʵĤ��λ

if ~isempty(psp) 
    V0 =  w' * psp; % ����˥���ͷ���Ӱ���µ������嵽�������Ĥ��λ�仯
else
    return;
end

E_k = zeros(No,1); % ģ��˥����������ʱ��˥���ͷ������������Ĥ��λ�½�
count = 1;
for k = 1: length(Timeline) % ����ÿ��ʱ�̣�(40)
    temp_cj = zeros(No,1); % kʱ��ÿ��Ԫ��������������¼
    
    E_k = E_k * factor;
    Vo(:,k) = V0(:,k) - E_k; % Ĥ��λ��ʱ��˥����Vo(:,k)��kʱ��Ĥ��λ
    row = find(Vo(:,k) >= theta); % kʱ�̷����������Ԫ
%     count = 1;
    while ~isempty(row) % ����ÿ�������Ĥ��λ��Ӱ��
        temp_cj(row) = temp_cj(row)+1;
        if length(theta) == No % ���ڶ�̬��ֵʱ�����ݲ�ͬ��ֵ�������
            E_k(row) = E_k(row) + theta(row);      
        else
%             E_k(row) = E_k(row) + theta;   %***
            E_k(row) = E_k(row) + Vo(row,k);
        end
        temp = Inf(No,1);
        temp(row) = Timeline(k); % kʱ�����巢��������������ֱ��¼��
        to2 = [to2 temp];  % ��ʱ�̸���Ԫ�ķ��������¼
        
        Vo(:,k) = V0(:,k) - E_k;                             
        row = find(Vo(:,k) >= theta);
        
%         count
        count = count + 1;
    end
    temp = Inf(No,1);
    temp(temp_cj > 0) = Timeline(k); % kʱ�̷�����������Ԫ��¼
    to1 = [to1 temp];
    
    cj = [cj temp_cj]; % ��ʱ�̸���Ԫ��������������¼
end

if ~isempty(to1)
    to1 = sort(to1,2); to2 = sort(to2,2);
    to1(:, all(to1 == Inf,1)) = []; to2(:, all(to2 == Inf,1)) = []; % ���ȫinf��  
    outNum = sum(cj,2);
end