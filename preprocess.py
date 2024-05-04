import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch


# Helper function to preprocess image into tensor from image path
def load_image_as_tensor(im_path, l=256, crop=False):
    if crop:
        preprocess = transforms.Compose([
            transforms.Resize(l),
            transforms.CenterCrop(l),
            transforms.ToTensor(),
            # # These normalization parameters are required by PyTorch's pre-trained VGG19
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((l,l)),
            transforms.ToTensor(),
            # # These normalization parameters are required by PyTorch's pre-trained VGG19
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    img = Image.open(im_path)
    img = img.convert('RGB')
    return preprocess(img).unsqueeze(0), img.size
    
def save_tensor_as_image(x, shape, path):
    # Denormalize the output image
    output_img = x.clone().squeeze()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cpu()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cpu()
    output_img = output_img.mul(std).add(mean).clamp(0, 1)
    # Convert tensor to PIL image
    output_img = transforms.ToPILImage()(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
    output_img = transforms.Resize((shape[1], shape[0]))(output_img)
    
    # Save or display the output image
    output_img.save(path)
    print(f"Saved image to {path}.")
    
# dataset class implementing desired transforms and normalizations
class MyDataset(Dataset):
    def __init__(self, relative_root):
        self.relative_root = relative_root
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.relative_root, self.data[idx])
        preprocessed_img, size = load_image_as_tensor(image_path)  # Implement load_image_as_tensor function
        return preprocessed_img, size

    def load_data(self):
        data = []
        for path in tqdm(os.listdir(self.relative_root)):
            data.append(path)
        return data


    

    
