import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights
from torchinfo import summary
from torch.nn.functional import mse_loss
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os, sys

sys.path.append("..")
from transfer_vgg_model import VGG19
from preprocess import load_image_as_tensor

# Implements algorithm adapted from Chen and Schmidt, Fast Patch-based Transfer of Arbitrary Style (2016).

# Efficient implementation from https://github.com/irasin/Pytorch_Style_Swap/blob/master/style_swap.py
def style_swap(content_feature, style_feature, kernel_size, stride=1):
    # content_feature and style_feature should have shape as (1, C, H, W)
    # kernel_size here is equivalent to extracted patch size

    # extract patches from style_feature with shape (1, C, H, W)
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride

    patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)

    # calculate Frobenius norm and normalize the patches at each filter
    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)

    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)

    # calculate the argmax at each spatial location, which means at each (kh, kw),
    # there should exist a filter which provides the biggest value of the output
    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    # deconv/transpose conv
    deconv_out = F.conv_transpose2d(one_hots, patches)

    # calculate the overlap from deconv/transpose conv
    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    # average the deconv result
    res = deconv_out / overlap
    return res

def transfer(content_im_path, style_im_path, layer_num, patch_size, stride=1, num_iter=500, _lambda=1e-3):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    content_im, content_im_shape = load_image_as_tensor(content_im_path, l=256)
    content_im = content_im.to(device)
    # Use cropping instead of rescaling to preserve spatial resolution of style
    style_im, style_im_shape = load_image_as_tensor(style_im_path, l=256, crop=True) 
    style_im = style_im.to(device)
    print(content_im.shape, style_im.shape)

    vgg19 = VGG19(content_layers=[layer_num], style_layers=None)
    vgg19.to(device)

    phi_C = vgg19(content_im)[0][1]
    phi_S = vgg19(style_im)[0][1]
    print(phi_C.shape, phi_S.shape)
    target_phi_ss = style_swap(phi_C, phi_S, patch_size, stride=stride)

    x = content_im.clone().to(device)
    optimizer = optim.Adam([x.requires_grad_()], lr=0.05)

    NUM_ITER = num_iter
    LAMBDA = _lambda
    for epoch in tqdm(range(NUM_ITER), desc=f"stylizing {content_im_path}"):
        optimizer.zero_grad()
        phi_x = vgg19(x)[0][1]
        
        # Style Loss
        L_stylize = torch.sum(torch.square(phi_x - target_phi_ss)) # Frobenius norm
        
        # Total Variation Regularization for smooth images
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        L_tv = (diff_i + diff_j)
        
        loss = L_stylize + LAMBDA*L_tv
        loss.backward()
        optimizer.step()
    
    return x.cpu().detach(), content_im_shape

if __name__ == "__main__":
    content_im_path = "../data/content-images/nature.jpg"
    style_im_path = "../data/style-images/vangogh-starry-night.jpg"
    transfer(content_im_path, style_im_path, layer_num=11, patch_size=4,stride=3, num_iter=900, _lambda=1e-2)

