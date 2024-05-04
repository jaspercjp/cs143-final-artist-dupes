import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.vgg import VGG19_Weights
from torchinfo import summary
from torch.nn.functional import mse_loss
from tqdm import tqdm
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from transfer_vgg_model import VGG19
from preprocess import load_image_as_tensor

def transfer(content_im_path, style_im_path, content_layers=[8], style_layers=[8,13,20,29], num_iter=3000, alpha=1, beta=1e5):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    content_im, content_im_shape = load_image_as_tensor(content_im_path)
    content_im = content_im.to(device)
    style_im, style_im_shape = load_image_as_tensor(style_im_path)
    style_im = style_im.to(device)
    vgg19 = VGG19(content_layers=content_layers, style_layers=style_layers)
    vgg19.to(device)

    target_F, _ = vgg19(content_im)
    _, target_G = vgg19(style_im)

    # Initialize with original content image
    x = content_im.clone()

    # Optimizer
    optimizer = optim.Adam([x.requires_grad_()], lr=0.05)

    # Optimization loop to perform gradient descent on the output image x
    for step in tqdm(range(num_iter), desc='Trasferring Style.'):
        def closure():
            optimizer.zero_grad()
            F, G = vgg19(x)# List of F's (as in Gaty's original paper), # List of G(ram matrice)'s 
            L_content = 0
            L_style = 0
           
            # this option uses mse loss which is elemnt-wise and most similar to gatys et.al
            for l in range(len(target_F)):
                L_content += mse_loss(F[l][1], target_F[l][1])
            for l in range(len(target_G)):
                L_style += mse_loss(G[l][1], target_G[l][1])

            loss = alpha*L_content + beta*L_style
            loss.backward()
            
            # print(f"Iter={step} | content loss={alpha*L_content.item()} | style loss={beta*L_style.item()}")
            return loss
        optimizer.step(closure)
        
    return x.cpu().detach()

if __name__ == "__main__":
    content_im_path = "../data/content-images/nature.jpg"
    style_im_path = "../data/style-images/vangogh-starry-night.jpg"
    transfer(content_im_path, style_im_path, content_layers=[3], style_layers=[8,13,20], num_iter=1000, beta=1e7)
