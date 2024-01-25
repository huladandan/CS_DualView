function [predict_im_1, predict_im_2,MSE_im_1,MSE_im_2] = Enc_main(ori_im_1,ori_im_2, measure_1, measure_2)

%% measurement matrix
A_1 = measure_1.A;
At_1 = measure_1.At;
A_2 = measure_2.A;
At_2 = measure_2.At;

%% measure
y = A_1(ori_im_1)+A_2(ori_im_2);
measure = measure_1;        
%%

AMP_iters = 15;
[predict_im_1,predict_im_2,~,MSE_sum]  = Prox_Moment_DualView(y',AMP_iters,A_1,At_1,A_2,At_2,measure_1,measure_2,measure.iterative_way);
MSE_im_1 = MSE_sum(15,3);
MSE_im_2 = MSE_sum(15,4);

end