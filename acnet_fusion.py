import torch
import os
from model import ScaleDense
import numpy as np


def _fuse_kernel(kernel, gamma, std):
    b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1, 1))
    b_gamma = np.tile(b_gamma, (kernel.shape[1], kernel.shape[2], kernel.shape[3], kernel.shape[4]))
    b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1, 1))
    b_std = np.tile(b_std, (kernel.shape[1], kernel.shape[2], kernel.shape[3], kernel.shape[4]))
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                                        square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w] += asym_kernel


def convert_deploy_weight(train_model, root_path):
    model_key = train_model.state_dict()
    square_conv_var_names = [name for name in model_key if SQUARE_KERNEL_KEYWORD in name]
    deploy_dict = {}

    for square_name in square_conv_var_names:
        square_kernel = model_key[square_name]
        square_mean = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv1.1.running_mean')]
        square_std = np.sqrt(model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv1.1.running_var')] + 1e-6)
        square_gamma = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv1.1.weight')]
        square_beta = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv1.1.bias')]

        x_kernel = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv2.0.weight')]
        x_mean = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv2.1.running_mean')]
        x_std = np.sqrt(model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv2.1.running_var')] + 1e-6)
        x_gamma = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv2.1.weight')]
        x_beta = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv2.1.bias')]

        y_kernel = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv3.0.weight')]
        y_mean = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv3.1.running_mean')]
        y_std = np.sqrt(model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv3.1.running_var')] + 1e-6)
        y_gamma = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv3.1.weight')]
        y_beta = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv3.1.bias')]

        z_kernel = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv4.0.weight')]
        z_mean = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv4.1.running_mean')]
        z_std = np.sqrt(model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv4.1.running_var')] + 1e-6)
        z_gamma = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv4.1.weight')]
        z_beta = model_key[square_name.replace(SQUARE_KERNEL_KEYWORD, 'conv4.1.bias')]

        fused_bias = square_beta + x_beta + y_beta + z_beta - square_mean * square_gamma / square_std \
                        - x_mean * x_gamma / x_std - y_mean * y_gamma / y_std - z_mean * z_gamma / z_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(x_kernel, x_gamma, x_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(y_kernel, y_gamma, y_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(z_kernel, z_gamma, z_std))
        

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        # deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k,v in model_key.items():
        if 'conv1' not in k and 'conv2' not in k and 'conv3' not in k and 'conv4' not in k:
            deploy_dict[k] = v


    torch.save(deploy_dict, root_path + 'deploy_weight.pth.tar')
    # deploy_model.load_state_dict(torch.load(root_path +'deploy_weight.pth.tar'))

if __name__ == "__main__":
    root_path = './pretrained_model/ScaleDense/'
    train_model = ScaleDense.ScaleDense(8, 5, deploy=False)
    train_model.load_state_dict
    (torch.load(root_path + "ScaleDense_best_model.pth.tar")['state_dict'])
    for param_tensor in train_model.state_dict():
            #打印 key value字典
        print(param_tensor,'\t',train_model.state_dict()[param_tensor].size())
    SQUARE_KERNEL_KEYWORD = 'conv1.0.weight'
    deploy_model = ScaleDense.ScaleDense(8, 5, deploy=True)
    convert_deploy_weight(train_model,  root_path)