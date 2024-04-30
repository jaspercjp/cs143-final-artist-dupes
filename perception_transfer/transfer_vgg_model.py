import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.vgg import VGG19_Weights
# from torchinfo import summary
from torch.nn.functional import mse_loss

from tqdm import tqdm
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Calculates the Gram matrix used to compute style score
def Gram(features):
    FF = features.clone()
    batch_size, channels, height, width = features.size()
    FF = FF.view(batch_size * channels, height * width)
    g_matrix = torch.mm(FF, FF.t())
    return g_matrix.div(batch_size * channels * height * width)

# Class for the Neural Transfer model based on VGG19 (pre-trained)
class NTVGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', weights=VGG19_Weights.DEFAULT)
        self.vgg19.eval()
        
        # Freeze the weights of VGG19
        for p in self.vgg19.parameters():
            p.requires_grad_(False)
            
        # Separate the CNN from the classification head
        self.features = nn.ModuleList(list(self.vgg19.features)[:37]).eval()
    
    # Pass the input through and record the layer responses.
    def forward(self, x, input_type='content'):
        F = []
        G = []
        for i,layer in enumerate(self.features):
            x = layer(x)
            # TODO: Replace these numbers with a list of layers to record as an input to the model
            if i==17: # Choose layer(s) to get the content representation from
                # print(x.shape)
                F.append((i, x))
            elif i in [11,13,15]: # Choose layers to get style representations from
                G.append((i, Gram(x)))
        return F, G

# Helper function to preprocess image into tensor from image path
def load_image_as_tensor(im_path, l=256):
    preprocess = transforms.Compose([
        transforms.Resize(l),
        transforms.ToTensor(),
        # These normalization parameters are required by PyTorch's pre-trained VGG19
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(im_path)
    return preprocess(img).unsqueeze(0), img.size

# # NOTE: transforms.Resize only resizes the smaller edge. Should we 
# # 	make the images have the exact same dimensions?
# content_im, content_im_shape = load_image_as_tensor("sample-nature.jpg")
# content_im = content_im.to(device)
# style_im, style_im_shape = load_image_as_tensor("sample-van-gogh.jpg", l=200)
# style_im = style_im.to(device)
# print("Image Shapes:", content_im.shape, style_im.shape)
# print("Content Image Mean:", content_im.max())
# model = NTVGG19()
# model.to(device)

# #summary(model, content_im.shape)
# #exit()

# target_F, _ = model(content_im, input_type='content')
# _, target_G = model(style_im, input_type='style')

# # This is the image we perform gradient descent on. We can either use
# # 1) The content image
# # 2) White noise
# x = content_im.clone()
# # x = torch.randn(content_im.shape).to(device)

# # Optimizer
# optimizer = optim.Adam([x.requires_grad_()], lr=0.01)

# # Optimization loop to perform gradient descent on the output image x
# num_steps = 2000
# alpha = 1
# beta = 1e9
# for step in tqdm(range(num_steps), desc='Trasferring Style.'):
#     def closure():
#         # x.data.clamp_(0, 1)
#         optimizer.zero_grad()
#         F, G = model(x)# List of F's (as in Gaty's original paper), # List of G(ram matrice)'s 
#         L_content = 0
#         L_style = 0
        

#         # this option uses mse loss which is elemnt-wise and most similar to gatys et.al
#         # for l in range(len(target_F)):
#         #     L_content += mse_loss(F[l][1], target_F[l][1])
#         # for l in range(len(target_G)):
#         #     L_style += mse_loss(G[l][1], target_G[l][1], reduction='mean').div(len(target_G))

#         # this option uses euclidian and frobenius norms, implementing Johnson et al's method
#         for l in range(len(target_F)):
#             CHW = target_F[l][1].shape[1] * target_F[l][1].shape[2] * target_F[l][1].shape[3]
#             L_content += (torch.norm(F[l][1] - target_F[l][1], p = 2) ** 2) / CHW

#         for l in range(len(target_G)):
#             CHW = target_F[l][1].shape[1] * target_F[l][1].shape[2] * target_F[l][1].shape[3]
#             L_style += (torch.norm(G[l][1] - target_G[l][1], p = 'fro') ** 2) 

#         loss = alpha*L_content + beta*L_style
#         loss.backward()
#         #print(f"Iter={step} | content loss={alpha*L_content.item()}")
#         print(f"Iter={step} | content loss={alpha*L_content.item()} | style loss={beta*L_style.item()}")
#         return loss
#     optimizer.step(closure)
    
# # Denormalize the output image
# output_img = x.clone().squeeze()
# mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
# std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
# print(output_img.shape, torch.max(output_img).item(), torch.min(output_img).item())
# output_img = output_img.mul(std).add(mean).clamp(0, 1)
# # Convert tensor to PIL image
# # plt.imshow(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
# output_img = transforms.ToPILImage()(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
# output_img = transforms.Resize((content_im_shape[1], content_im_shape[0]))(output_img)
# # Save or display the output image
# output_img.save("no-clamp-training.jpg")

