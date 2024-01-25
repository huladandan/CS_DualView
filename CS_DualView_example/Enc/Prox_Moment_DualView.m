function [x_hat_1,x_hat_2,PSNR_sum,MSE_sum] = Prox_Moment_DualView(y,iters,M_func_1,Mt_func_1,M_func_2,Mt_func_2,measure_1,measure_2,iterative_way)
global global_time
global_time = 0;
randn('state',0);
rand('state',0);

PSNR_func = @(x_hat, ori_im) PSNR(abs(ori_im),abs(x_hat));
m = length(y);
alpha = 1;

%% 初始化参数
switch iterative_way
    case 1        % dual_view_model
        M_1=@(x) M_func_1(x);
        Mt_1=@(z) Mt_func_1(z);
        M_2=@(x) M_func_2(x);
        Mt_2=@(z) Mt_func_2(z);
        denoi_1=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure_1.image_width,measure_1.image_height,measure_1.denoize_name);
        denoi_2=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure_2.image_width,measure_2.image_height,measure_2.denoize_name);

        %%
        x_t_1{2} = zeros(measure_1.length,1);
        x_t_2{2} = zeros(measure_2.length,1);

        %% 对y = y1+y2 直接进行重建出两个x（观测矩阵不同）
        tic
        x_t_1{1}=Mt_1(y);
        x_t_2{1}=Mt_2(y);
        global_time = global_time + toc;

        %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
        %%
        sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
        sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
        
        x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1));
        x_t_2{2}=double(denoi_2(x_t_2{1},sigma_hat_2));

        %%
        v_t_1 = zeros(measure_1.length,1);
        v_t_2 = zeros(measure_2.length,1);
        PSNR_sum=zeros(iters,4);
        MSE_sum=zeros(iters,4);
        y_error = zeros(iters,1);
        sigma_sum = zeros(iters,2);
        % iterative_way = 1;

        epsilon = 1;
        tic
        eta_1=randn(1,measure_1.length);
        eta_2=randn(1,measure_2.length);
        gamma_1=1/(m*epsilon).*eta_1*(denoi_1(x_t_1{1}+epsilon*eta_1',sigma_hat_1)-x_t_1{2});
        gamma_2=1/(m*epsilon).*eta_2*(denoi_2(x_t_2{1}+epsilon*eta_2',sigma_hat_2)-x_t_2{2});
        global_time = global_time + toc;

       
        
end
%%

%% 迭代
for i=1:iters
    
    
    switch iterative_way
        case 1
            tic
            v_temp = y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))';
            global_time = global_time + toc;
            gamma_1=1/(m*epsilon).*eta_1*(denoi_1(x_t_1{1}+epsilon*eta_1',sigma_hat_1)-x_t_1{2});
            gamma_2=1/(m*epsilon).*eta_2*(denoi_2(x_t_2{1}+epsilon*eta_2',sigma_hat_2)-x_t_2{2});
            tic
            gamma = gamma_1+gamma_2;
            
            v_t_1=gamma.*v_t_1+Mt_1(v_temp);
            x_t_1{1}=x_t_1{2}+alpha.*v_t_1;
            global_time = global_time + toc;
            sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
            
            tic
            v_t_2=gamma.*v_t_2+Mt_2(v_temp);
            x_t_2{1}=x_t_2{2}+alpha.*v_t_2;
            global_time = global_time + toc;
            sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
            
            x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1));
            x_t_2{2}=double(denoi_2(x_t_2{1},sigma_hat_2));

    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_error(i) = sum(abs(v_temp)/m);
    %%
    sigma_sum(i,1) = sigma_hat_1;
    sigma_sum(i,2) = sigma_hat_2;
    I1 = reshape(x_t_1{1},measure_1.image_height,measure_1.image_width);    %图1未去噪（{2}是{1}的去噪结果
    I2 = reshape(x_t_2{1},measure_2.image_height,measure_2.image_width);    %图2未去噪   
    I3 = reshape(x_t_1{2},measure_1.image_height,measure_1.image_width);    %图1去噪  
    I4 = reshape(x_t_2{2},measure_2.image_height,measure_2.image_width);    %图2去噪  
    [PSNR_sum(i,1), MSE_sum(i,1)] = PSNR_func(I1, measure_1.ori_im);        %第一列为图1未去噪
    [PSNR_sum(i,2), MSE_sum(i,2)] = PSNR_func(I2, measure_2.ori_im);        %第二列为图2未去噪
    [PSNR_sum(i,3), MSE_sum(i,3)] = PSNR_func(I3, measure_1.ori_im);        %第三列为图1去噪
    [PSNR_sum(i,4), MSE_sum(i,4)] = PSNR_func(I4, measure_2.ori_im);        %第四列为图2去噪

    PSNR_sum(i,5) = PSNR_sum(i,3)+PSNR_sum(i,4);
end


x_hat_1=reshape(x_t_1{2},[measure_1.image_height measure_1.image_width]);
x_hat_2=reshape(x_t_2{2},[measure_2.image_height measure_2.image_width]);
[PSNR_sum(i,3), MSE_sum(i,3)] = PSNR_func(x_hat_1, measure_1.ori_im);
[PSNR_sum(i,4), MSE_sum(i,4)] = PSNR_func(x_hat_2, measure_2.ori_im);


end


