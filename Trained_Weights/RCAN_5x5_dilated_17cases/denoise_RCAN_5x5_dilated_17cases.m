function output=denoise_RCAN_5x5_dilated_17cases(noisy, sigma_hat)
global global_time
if sigma_hat == 15
    name='RCAN_5x5_dilated_sigma15to15.mat';
elseif sigma_hat == 25
    name='RCAN_5x5_dilated_sigma25to25.mat';
elseif sigma_hat == 50
    name='RCAN_5x5_dilated_sigma50to50.mat';
elseif sigma_hat<=5
    name='rcan_5x5_dilated_17cases_sigma0to5.mat';
elseif sigma_hat>5&&sigma_hat<=10
    name='rcan_5x5_dilated_17cases_sigma5to10.mat';
elseif sigma_hat>10&&sigma_hat<=15
    name='rcan_5x5_dilated_17cases_sigma10to15.mat';
elseif sigma_hat>15&&sigma_hat<=20
    name='rcan_5x5_dilated_17cases_sigma15to20.mat';
elseif sigma_hat>20&&sigma_hat<=30
    name='rcan_5x5_dilated_17cases_sigma20to30.mat';
elseif sigma_hat>30&&sigma_hat<=40
    name='rcan_5x5_dilated_17cases_sigma30to40.mat';
elseif sigma_hat>40&&sigma_hat<=50
    name='rcan_5x5_dilated_17cases_sigma40to50.mat';
elseif sigma_hat>50&&sigma_hat<=60
    name='rcan_5x5_dilated_17cases_sigma50to60.mat';
elseif sigma_hat>60&&sigma_hat<=70
    name='rcan_5x5_dilated_17cases_sigma60to70.mat';
elseif sigma_hat>70&&sigma_hat<=80
    name='rcan_5x5_dilated_17cases_sigma70to80.mat';
elseif sigma_hat>80&&sigma_hat<=90
    name='rcan_5x5_dilated_17cases_sigma80to90.mat';
elseif sigma_hat>90&&sigma_hat<=100
    name='rcan_5x5_dilated_17cases_sigma90to100.mat';
elseif sigma_hat>100&&sigma_hat<=125
    name='rcan_5x5_dilated_17cases_sigma100to125.mat';
elseif sigma_hat>125&&sigma_hat<=150
    name='rcan_5x5_dilated_17cases_sigma125to150.mat';
elseif sigma_hat>150&&sigma_hat<=300
    name='rcan_5x5_dilated_17cases_sigma150to300.mat';
elseif sigma_hat>300&&sigma_hat<=500
    name='rcan_5x5_dilated_17cases_sigma300to500.mat';
elseif sigma_hat>500&&sigma_hat<=1000
    name='rcan_5x5_dilated_17cases_sigma500to1000.mat';
else
    name='rcan_5x5_dilated_17cases_sigma500to1000.mat';
end

useGPU = 1;
load(name);%loads net
net.maintain_layer=[2,5,12,15,22,25,32,35,42,45,52,55,62,65,72,75,84,87,94,97,104,107,114,117,124,127,134,137,144,147,154,157];
patch_flag = 0;
%P_scale = max(max(noisy));%%revised why?
P_scale = 255;
if patch_flag == 1
    img_size = size(noisy);
    patch_size = [128, 128];
    step_size = [4,4];
    batch_size = 2;
    noisy = im2patch(noisy,patch_size,step_size);
    noisy1(:,:,1,:) = noisy;
    noisy = noisy1;
    input_list_index = 1:batch_size:size(noisy,4);
    
    if useGPU == 1
        
        noisy = single(noisy/(P_scale/2)-1);
        noisy=real(noisy);
        input = gpuArray(noisy);
        net = vl_simplenn_move(net, 'gpu') ;
        tic
        for i = 1:length(input_list_index)
            input_list = (i-1)*batch_size+1:min((i-1)*batch_size +batch_size,size(noisy,4));
            res = vl_simplenn(net,input(:,:,:,input_list),[],[],'conserveMemory',true,'mode','test');
            output(:,:,:,input_list) = res(end).x;
        end
        global_time = global_time+toc;
        output = double(gather((output+1)*P_scale/2));
    else
        
        noisy = noisy/(P_scale/2)-1;
        input = noisy;
        res = simplenn_matlab(net, input);
        output = res(end).x;
        output = (output+1)*P_scale/2;
        
    end
    output = squeeze(output);
    output = patch2im(output, img_size, patch_size, step_size);
else
    if useGPU == 1
        
        noisy = single(noisy/(P_scale/2)-1);
        noisy=real(noisy);
        input = gpuArray(noisy);
        
        net = vl_simplenn_move(net, 'gpu') ;
        tic
        res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        output = res(end).x;
        global_time = global_time+toc;
        output = double(gather((output+1)*P_scale/2));
    else
        
        noisy = noisy/(P_scale/2)-1;
        input = noisy;
        res = simplenn_matlab(net, input);  
        output = res(end).x;
        output = (output+1)*P_scale/2;
        
    end
end

end
