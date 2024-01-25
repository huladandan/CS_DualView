function [measure,quantize] = para_set_individual(measure, denoize_choice, Test_image_dir, seed)

measure.seed = seed;
randn('state',seed);
rand('state',seed);
measure.Test_image_dir = Test_image_dir;
measure.ori_im = double(imread(Test_image_dir));
quantize = [];

%% measurement parameter setting
if strcmp(measure.Test_set_name, 'urban100')

	[image_height, image_width]=size(measure.ori_im);
	[measure.block_height,measure.block_width, image_height, image_width] = blocksize_urban100_set(image_height, image_width);
	measure.ori_im = measure.ori_im(1:image_height, 1:image_width);
else
    measure.block_width = 64;
    measure.block_height = 64;
end

[image_height, image_width]=size(measure.ori_im);
measure.image_height = image_height;
measure.image_width = image_width; % divide the whole image into small blocks
measure.P_image=randperm(measure.image_height*measure.image_width);
measure.P_block=randperm(measure.block_width*measure.block_height);
measure.length = measure.image_width*measure.image_height;


%%
denoise_name_all = {'DnCNN_20Layers_3x3_17cases','RCAN_5x5_dilated_17cases', 'Restormer'};

measure.denoize_name = denoise_name_all{denoize_choice};
measure.model = 'Bernoulli';

end
