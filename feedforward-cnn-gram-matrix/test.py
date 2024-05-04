import numpy as np
import torch 
from image_transform_net import ImageTransformerRef
from preprocess import load_image_as_tensor
#from transfer_vgg_model import load_image_as_tensor
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from residual_block import ResidualBlock

device = torch.device("cpu")
test_image = "../data/00000002_(2).jpg"
# test_image = "sample-nature.jpg"
# test_image = "../sample-monet.jpg"
# test_image = "../sample-van-gogh.jpg"

def main():
    model = ImageTransformerRef().to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    test_tensor, shape = load_image_as_tensor(test_image, l=256)
    out = model(test_tensor)

    # denormalizing the output image
    output_img = torch.squeeze(out)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to("cpu")
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to("cpu")
    output_img = output_img.mul(std).add(mean)
    print(output_img.shape, torch.max(output_img).item(), torch.min(output_img).item())
    output_img = output_img.clamp(0, 1)

    output_img = transforms.ToPILImage()(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
    output_img = transforms.Resize((shape[1], shape[0]))(output_img)
    # Save or display the output image
    output_img.save("test-stylized.jpg")

    print(out.shape)

main()
