import numpy as np
import torch 
from image_transform_net import ImageTransformer
from transfer_vgg_model import load_image_as_tensor
from PIL import Image
from torchvision import transforms

test_image = "sample-nature.jpg"

def main():
    model = ImageTransformer()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    test_tensor, shape = load_image_as_tensor(test_image)
    print(test_tensor.shape)
    print(torch.max(test_tensor))
    print(torch.min(test_tensor))
    out = model(test_tensor)

    # out = (out - torch.min(out))
    # out = out / torch.max(out)
    # out = out * 255
    # denormalizing the output image
    output_img = torch.squeeze(out)
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to("cpu")
    # std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to("cpu")
    # print(output_img.shape, torch.max(output_img).item(), torch.min(output_img).item())
    # output_img = output_img.mul(std).add(mean).clamp(0, 1)

    output_img = transforms.ToPILImage()(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
    output_img = transforms.Resize((shape[1], shape[0]))(output_img)
    # Save or display the output image
    output_img.save("no-clamp-training.jpg")

    print(out.shape)

main()