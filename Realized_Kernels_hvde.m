function [MKer H] = Realized_Kernels_hvde(data, time, mJit, kernel)
%
% Realized covariance/variance based on flat-top (Tukey-Hanning) or non 
% flat-top (parzen) realized kernel
%
% INPUT: 
% 
% data: 1xN cell containing prices for N stocks OR Tx1 vector for a univariate kernel 
% time: same structure
% mJit: Amount of Jittering
% kernel: choice of kernel function
%
% OUTPUT:
% MKer: Multivariate or univariate kernel measure 
% H: Bandwidth used in the kerl
%
% Harry Vander Elst (May 2014)
%
if iscell(data); dim = 'multi';else dim = 'uni'; end
if nargin<4; kernel = 'parzen'; end
if nargin<3; mJit=2; end

if strcmp(dim,'multi')
NtS = Noise_to_Signal(data, time);
[NewData NewTimes] = hvde_refresh_time(data,time);
tm = NewTimes(:,1);
Pr = NewData;
else
NtS = Noise_to_Signal({data}, {time});
tm = time;
Pr = data;
end
    
timrg = max(tm)-min(tm); % length of periods in seconds
dimPr = size(Pr,2);      % get the number of price series

% Jitter first m and last m observations
jitPr = [ mean(Pr(1:mJit,:),1); Pr(mJit+1:end-mJit,:); mean(Pr(end-mJit+1:end,:),1)];

rets  = 100*diff(log(jitPr));%100*diff(log(jitPr));
nRt   = size(rets,1);

% Get scaling factor
if (mJit>1)
    jittm  = [ tm(1:mJit), fliplr(tm(end-mJit+1:end))];
    sc     = timrg/(timrg-(mJit-1:-1:1)/mJit*sum(diff(jittm),2));
else
    sc = 1;
end;

H = get_bandwidth(NtS, nRt, kernel);
H = min([H nRt-1]);

weights = get_weights(H, kernel);

Mker=0;
for h=1:H+1
    gamma = rets'*[ rets(h:nRt,:); zeros(h-1,dimPr)];
    Mker = Mker + weights(h)*(gamma+gamma');
end;

% return average of subsampled kernels
MKer = Mker*sc;

end
% ======================================================================= %
% Aux. function to compute the noise to signal ratio based on subsampled RV
% ======================================================================= %
function NtS = Noise_to_Signal(data, time)

cross = length(data);
new_data = get_subsamples(data,time,20*60,10);
Sub = size(new_data,1);
NtoS = zeros(cross,1);

for cpt=1:cross
Ret = diff(100*log(data{cpt}));
nNonZero = sum(~Ret==0);
DenseRV = Ret'*Ret;

Sub_SpaRV = zeros(Sub,1);

for cpt2=1:Sub
SparseRet = diff(100*log(new_data{cpt2,cpt}));
Sub_SpaRV(cpt2,1) = SparseRet'*SparseRet;
end
SparseRV = mean(Sub_SpaRV);
NtoS(cpt,1) = (DenseRV/(2*nNonZero))/SparseRV;

end
NtoS(isinf(NtoS))=1;
NtS = mean(NtoS);

end
% ======================================================================= %
% Aux. function to compute the optimal bandwidth
% ======================================================================= %
function H = get_bandwidth(NtS, nRt, kernel)
%
% NtS: square of Noise-to-signal ratio
% nRt: Amount of data after jittering
% kernel: choice of kernel
%
kernel = lower(kernel);

if strcmpi(kernel,'th2') || strcmpi(kernel,'th') || strcmpi(kernel,'tukeyhanning')
% Tukey–Hanning_2 is the optimal choice for flat-to kernel BNHLS (2008)
    cStar = 5.74;
    % H ~ c Xi N^(1/2)
    H = cStar * (NtS)^(1/2) * nRt^(1/2);
elseif strcmpi(kernel,'parzen') || strcmpi(kernel,'par') || strcmpi(kernel,'p')
% Parzen is the optimal choice for non flat-top kernel as pointed in BNHLS (2011)
    cStar = ((12)^2/0.269)^(1/5);
    % H ~ c Xi^(4/5) N^(3/5)
    H = cStar * (NtS)^(2/5) * nRt^(3/5);
else
    error('Wrong choice of kernel function')
end

H = ceil(H);
end
% ======================================================================= %
% Aux. function to compute the weights - weights \in R^(H+1x1)
% ======================================================================= %
function weights = get_weights(H, kernel)

kernel = lower(kernel);
x = (1:H)';

if H>0
%**************************************************************************
if strcmpi(kernel,'th2') || strcmpi(kernel,'th') || strcmpi(kernel,'tukeyhanning')
% Tukey–Hanning_2 is the optimal choice for flat-to kernel BNHLS (2008)
    x = (x-1)/H;
    weights = sin(pi/2*(1-x)).^2;
elseif strcmpi(kernel,'parzen') || strcmpi(kernel,'par') || strcmpi(kernel,'p')
% Parzen is the optimal choice for non flat-top kernel as pointed in BNHLS (2011)
    x = x./(H+1);
    weights = (1-6*x.^2+6*x.^3).*(x>=0 & x<=1/2) + 2*(1-x).^3.*(x>1/2 & x<1);
else
    error('Wrong choice of kernel function')
end
%**************************************************************************
else
    weights = [];
end

weights = [.5; weights];

end
% ----------------------------------------------------------------------- %
% ======================================================================= %
% Aux. function to put data in blocks
% ======================================================================= %
function [dataBis timeBis] = get_subsamples(data,time,freq,step)

if nargin<4; step = 1; end
if nargin<4; freq = 20*60; end
if mod(freq,step)~=0; error('frequency and step size should agree'); end

cross = length(data);

[Data Time] = hvde_tick_interpolation(data,time,step);

dataBis = cell(freq/step,cross);
timeBis = cell(freq/step,1);

Data_Cell = cell(1,cross);
Time_Cell = cell(1,cross);

for sckda=1:cross
    Data_Cell(sckda) = {Data(:,sckda)};
    Time_Cell(sckda) = {Time(:,sckda)};
end

cpt3=1;
for cpt1=step:step:freq
[new_data new_time] = hvde_tick_interpolation(Data_Cell,Time_Cell,freq,1,cpt1-step);
for cpt2=1:cross
dataBis{cpt3,cpt2} = new_data(:,cpt2);
timeBis{cpt3,cpt2} = new_time(:,cpt2);
end
cpt3=cpt3+1;
end


end