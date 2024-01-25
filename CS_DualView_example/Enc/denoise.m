function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
% function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
%DENOISE takes a signal with additive white Guassian noisy and an estimate
%of the standard deviation of that noise and applies some denosier to
%produce a denoised version of the input signal
% Input:
%       noisy       : signal to be denoised
%       sigma_hat   : estimate of the standard deviation of the noise
%       width   : width of the noisy signal
%       height  : height of the noisy signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%Output:
%       denoised   : the denoised signal.

%To apply additional denoisers within the D-AMP framework simply add
%aditional case statements to this function and modify the calls to D-AMP
global global_time

noisy=reshape(noisy,[height,width]);

switch denoiser
        
    case 'DnCNN_20Layers_3x3_17cases'
        output=denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat);
        
    case 'RCAN_5x5_dilated_17cases'
        output=denoise_RCAN_5x5_dilated_17cases(noisy, sigma_hat);
        
    case 'Restormer'
        if sigma_hat >= 150
            output=denoise_RCAN_3x3_dilated_17cases(noisy, sigma_hat);
        else
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.Restormer_denoise_matlab.denoiser(noisy, height, width, sigma_hat));
            output = reshape(output,width,height);
            output = output';
        end
        
    otherwise
        error('Unrecognized Denoiser')
end 
denoised=output(:);
end

