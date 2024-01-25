function output=denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat)
global global_time
if sigma_hat<=5
    name='DnCNN_20Layers_3x3_17cases_sigma0to5.mat';
elseif sigma_hat>5&&sigma_hat<=10
    name='DnCNN_20Layers_3x3_17cases_sigma5to10.mat';
elseif sigma_hat>10&&sigma_hat<=15
    name='DnCNN_20Layers_3x3_17cases_sigma10to15.mat';
elseif sigma_hat>15&&sigma_hat<=20
    name='DnCNN_20Layers_3x3_17cases_sigma15to20.mat';
elseif sigma_hat>20&&sigma_hat<=30
    name='DnCNN_20Layers_3x3_17cases_sigma20to30.mat';
elseif sigma_hat>30&&sigma_hat<=40
    name='DnCNN_20Layers_3x3_17cases_sigma30to40.mat';
elseif sigma_hat>40&&sigma_hat<=50
    name='DnCNN_20Layers_3x3_17cases_sigma40to50.mat';
elseif sigma_hat>50&&sigma_hat<=60
    name='DnCNN_20Layers_3x3_17cases_sigma50to60.mat';
elseif sigma_hat>60&&sigma_hat<=70
    name='DnCNN_20Layers_3x3_17cases_sigma60to70.mat';
elseif sigma_hat>70&&sigma_hat<=80
    name='DnCNN_20Layers_3x3_17cases_sigma70to80.mat';
elseif sigma_hat>80&&sigma_hat<=90
    name='DnCNN_20Layers_3x3_17cases_sigma80to90.mat';
elseif sigma_hat>90&&sigma_hat<=100
    name='DnCNN_20Layers_3x3_17cases_sigma90to100.mat';
elseif sigma_hat>100&&sigma_hat<=125
    name='DnCNN_20Layers_3x3_17cases_sigma100to125.mat';
elseif sigma_hat>125&&sigma_hat<=150
    name='DnCNN_20Layers_3x3_17cases_sigma125to150.mat';
elseif sigma_hat>150&&sigma_hat<=300
    name='DnCNN_20Layers_3x3_17cases_sigma150to300.mat';
elseif sigma_hat>300&&sigma_hat<=500
    name='DnCNN_20Layers_3x3_17cases_sigma300to500.mat';
elseif sigma_hat>500&&sigma_hat<=1000
    name='DnCNN_20Layers_3x3_17cases_sigma500to1000.mat';
else
    name='DnCNN_20Layers_3x3_17cases_sigma500to1000.mat';
end
useGPU = 1;
load(name);%loads net
net;
net.layers = net.layers(1:end-1);

if useGPU
    
    noisy = single(noisy/(255/2)-1);
    noisy=real(noisy);
    net = vl_simplenn_move(net, 'gpu') ;
    input = gpuArray(noisy);
    tic
    res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    output = input - res(end).x;
    Time_temp=toc;
    global_time = global_time+Time_temp;
    output = gather(output);
    output = (output+1)*255/2;
    
else
    
    noisy = noisy/255;
    input = noisy;
    net.layers = net.layers(1:end-1);
    res = simplenn_matlab(net, input);
    output = input - res(end).x;
    output = output*255;
    
end


end
