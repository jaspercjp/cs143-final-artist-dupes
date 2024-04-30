
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm


# Helper function to preprocess image into tensor from image path
def load_image_as_tensor(im_path, l=256):
    preprocess = transforms.Compose([
        transforms.Resize((l, l)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        # These normalization parameters are required by PyTorch's pre-trained VGG19
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(im_path)
    img = img.convert('RGB')
    return preprocess(img).unsqueeze(0), img.size


# dataset class
class LandscapeDataset(Dataset):
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



def main():
    relative_root = '../data/'
    dataset = LandscapeDataset(relative_root=relative_root)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle = True)

    with open("data.pkl", 'wb') as f:
        pickle.dump(dataloader, f)

main()


    

    