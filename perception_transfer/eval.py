from image_transform_net import ImageTransformer
from preprocess import load_image_as_tensor
import torch
import numpy as np
from torchvision import transforms
from torchinfo import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = ImageTransformer().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

x, content_im_shape = load_image_as_tensor("../sample-nature.jpg")
summary(model, x.shape)
print("Content Input Shape:", content_im_shape)
x = x.to(device)
x = model(x)

output_img = x.clone().squeeze()
#mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
#std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
#print(output_img.shape, torch.max(output_img).item(), torch.min(output_img).item())
#output_img = output_img.mul(std).add(mean).clamp(0, 1)
print(output_img.shape, torch.max(output_img).item(), torch.min(output_img).item())
# Convert tensor to PIL image
# plt.imshow(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
output_img = transforms.ToPILImage()(np.moveaxis(output_img.cpu().detach().numpy(), 0, -1))
# output_img = transforms.Resize((content_im_shape[1], content_im_shape[0]))(output_img)
# Save or display the output image
output_img.save("perceptual-test-2.jpg")
