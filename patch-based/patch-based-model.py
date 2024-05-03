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
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Style Swap Technique
# - Feature extraction by feeding into VGG 
# - Extract patches and match them between the style and content features based on nearest neighbor appraoch
# - Transform the content features back into the image space 

# Inverse Network Architecture - Mapping feature representations back into pixel space
# - Unlike Gatys's iterative optimiziation to convergence, Chen's approach use inverse network to mirror the encoder part of the VGG network 
# - Network is separate from the 


def style_swap(content_features, style_features, patch_size=3, stride=1):
    """
    Extract patches from style features and replace content feature patches with nearest matching style patches.
    """
    # Extract patches from style features
    style_patches = F.unfold(style_features, kernel_size=patch_size, stride=stride)
    style_patches = style_patches.permute(0, 2, 1)

    # Compute patch similarities and replace
    content_patches = F.unfold(content_features, kernel_size=patch_size, stride=stride)
    content_patches = content_patches.permute(0, 2, 1)

    similarity = torch.einsum('ijk,ilk->ijl', content_patches, style_patches)
    _, indices = similarity.max(dim=2)

    matched_patches = style_patches[torch.arange(style_patches.size(0)).unsqueeze(1), indices]
    matched_patches = matched_patches.permute(0, 2, 1)
    
    # Fold back to form the image
    output = F.fold(matched_patches, output_size=content_features.shape[-2:], kernel_size=patch_size, stride=stride)
    return output

###########################

# Fast Texture 
# - Coherent Synthesis - grows texture patches pixel by pixel in a scan-line order across the image 
# For each new pixel, the algo chooses the best match from a list of candidate pixels based on already synthesized pixels taken from the source image 

# Enhancements to coherent synthesis method
# - Introduces probability 'p' so a random candidate pixel from anywhere in the texture img is added to candidate list (to add randomness)
# - Expands search space by adding random candidates to address risks in flat looking images 
# - Modifies original image difference measure to include weighted combo of differences from both source and target image 

# Calculates the Gram matrix used to compute style score
def Gram(features):
    FF = features.clone()
    batch_size, channels, height, width = features.size()
    FF = FF.view(batch_size * channels, height * width)
    g_matrix = torch.mm(FF, FF.t())

    # Introducing randomness to the Gram matrix calculation 
    alpha = 0.5
    noise = torch.randn(g_matrix.size()).to(device) * alpha
    g_matrix += noise 
    return g_matrix.div(batch_size * channels * height * width)

class InverseVGG(nn.Module):
    # super(InverseVGG, self).__init__()
    # # First upsampling step
    # self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    # self.instancenorm1 = nn.InstanceNorm2d(128)
    # self.relu1 = nn.ReLU(inplace=True)
    
    # # Second upsampling step
    # self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest') # Nearest neihgbor upsampling
    # self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    # self.instancenorm2 = nn.InstanceNorm2d(128)
    # self.relu2 = nn.ReLU(inplace=True)
    
    # # Third upsampling step
    # self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    # self.instancenorm3 = nn.InstanceNorm2d(64)
    # self.relu3 = nn.ReLU(inplace=True)
    
    # # Fourth upsampling step
    # self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest') # Nearest neihgbor upsampling
    # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    # self.instancenorm4 = nn.InstanceNorm2d(64)
    # self.relu4 = nn.ReLU(inplace=True)
    
    # # Final convolution to get to 3 channels
    # self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def __init__(self):
        super(InverseVGG, self).__init__()
        # Upsampling and feature transformation 
        self.layers = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)
    
def list_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(list_files('train/train_data'))

class TruncatedVGG19(nn.Module):
    def __init__(self):
        super(TruncatedVGG19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        
        # Select layers up to relu3_1 which is layer index 12 in the pretrained VGG19 features
        self.features = nn.Sequential(*list(vgg_pretrained_features.children())[:12])

    def forward(self, x):
        x = self.features(x)
        return x


def train_inverse_network(inverse_model, style_im, vgg_model, train_loader, epochs, optimizer, criterion, device):
    inverse_model.train()
    vgg_model.eval()  

    for epoch in range(epochs):
        running_loss = 0.0
        for content_images, target_images in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            content_images, target_images = content_images.to(device), target_images.to(device)
            
            # Get VGG features
            with torch.no_grad():
                content_features = vgg_model(content_images)
                style_features = vgg_model(style_im)  
            
            swapped_features = style_swap(content_features, style_features)

            # Forward pass through the inverse model
            outputs = inverse_model(content_features)  # or swapped_features if using style swap
            
            print("Output shape:", outputs.shape)
            print("Target shape:", target_images.shape)

            # Finding loss
            loss = criterion(outputs, target_images)
            
            # Backprop and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Log the average loss per epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print("training done")

# Helper function to preprocess image into tensor from image path
def load_image_as_tensor(im_path, target_size=(256,256)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),  # Resize the smallest edge to target_size
        transforms.CenterCrop(target_size),  # Crop to make it square
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(im_path)
    return preprocess(img).unsqueeze(0), img.size
    
# NOTE: transforms.Resize only resizes the smaller edge. Should we 
# 	make the images have the exact same dimensions?
content_im, content_im_shape = load_image_as_tensor("sample-nature.jpg")
content_im = content_im.to(device)
style_im, style_im_shape = load_image_as_tensor("sample-van-gogh.jpg")
style_im = style_im.to(device)
print("Image Shapes:", content_im.shape, style_im.shape)
print("Content Image Mean:", content_im.max())


# Initialize models
inverse_model = InverseVGG().to(device)
vgg_model = TruncatedVGG19().to(device)

# Define loss function and optimizer
optimizer = optim.Adam(inverse_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

class ImageReconstructionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.file_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image  # return image both as input and target
        
# Data transformation and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageReconstructionDataset(root_dir='train/train_data', transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Apply style swap beteween content features and style features 
content_feats = vgg_model(content_im)
style_feats = vgg_model(style_im)
swapped_feats = style_swap(content_features=content_feats, style_features=style_feats)

# Train the network
train_inverse_network(inverse_model, style_im, vgg_model, train_loader, epochs=2, optimizer=optimizer, criterion=criterion, device=device)


# Save or display the final image
output_image = inverse_model(swapped_features).squeeze().detach()
output_image = transforms.functional.to_pil_image(output_image.clamp_(0, 1))
output_image.save("output.jpg")

# # This is the image we perform gradient descent on. We can either use
# # 1) The content image
# # 2) White noise
# # 3) Textured Noise 
# x = content_im.clone()
# x = torch.randn(content_im.shape).to(device)

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

