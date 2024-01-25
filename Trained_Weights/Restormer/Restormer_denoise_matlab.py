import argparse
import os

import numpy as np


import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer

import array



def denoiser(img_L, img_weight, img_height, noise_level_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # print(script_dir)
    sigma_test = noise_level_model
    # -----------------------------------------------model parameter setting
    parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising using Restormer')
    parser.add_argument('--weights', default=os.path.join(script_dir, 'pretrained_models', 'gaussian_gray_denoising'),
                        type=str, help='Path to weights')
    parser.add_argument('--model_type', default='blind', type=str,
                        help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
    args = parser.parse_args()

    ####### Load yaml #######
    if args.model_type == 'blind':
        yaml_file = os.path.join(script_dir, 'Options', 'GaussianGrayDenoising_Restormer.yml')
    else:
        yaml_file = os.path.join(script_dir, 'Options', f'GaussianGrayDenoising_RestormerSigma{args.sigmas}.yml')
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    s = x['network_g'].pop('type')
    # --------------------------------------------------model injection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_restoration = Restormer(**x['network_g'])
    if args.model_type == 'blind':
        weights = args.weights + '_blind.pth'
    else:
        weights = args.weights + '_sigma' + str(sigma_test) + '.pth'
    checkpoint = torch.load(weights)
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration = model_restoration.to(device)
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # --------------------------------------------------deat preprocess
    img_L = np.array(img_L)
    img_L = np.reshape(img_L, (int(img_weight), int(img_height)))
    if img_L.ndim == 2:
        img_L = np.expand_dims(img_L, axis=2)
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
    input_ = img_L.to(device)
    ##########################
    factor = 8
    # --------------------------------------------------model evaluation
    with torch.no_grad():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        # # 计时开始
        # start_time = time.time()

        img_E = model_restoration(input_)

        # # 计时结束
        # end_time = time.time()
        # # 创建csv文件记录时间
        # with open('time.csv', 'a') as f:
        #     f.write(str(end_time - start_time) + '\n')
        #     f.close()

        # Unpad images to original dimensions
    img_E = img_E[:, :, :h, :w]

    img_E = torch.clamp(img_E, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    img_E = img_E * 255.0
    img_E = np.reshape(img_E, (int(img_weight) * int(img_height), 1))
    img_E = array.array('d', img_E)
    return img_E
