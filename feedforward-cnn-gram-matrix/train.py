
import torch
from image_transform_net import ImageTransformerRef
from torch.optim import Adam, LBFGS
from preprocess import load_image_as_tensor, vgg_normalize, MyDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hyperparameters import EPOCHS
from torch.nn.functional import mse_loss
import torch.nn as nn
from torchvision import transforms
from datetime import datetime
import numpy as np
from torchinfo import summary 

import sys
sys.path.append("../")
from transfer_vgg_model import VGG19

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 2
relative_root = '../data/small-data'
dataset = MyDataset(relative_root=relative_root)
training_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle = True)

model = ImageTransformerRef()
model.to(device)
optimizer = Adam(model.parameters(), lr = 0.001)

loss_network = VGG19(content_layers=[1], style_layers=[3,8,13,20])
loss_network.to(device)

style_image, _ = load_image_as_tensor('../data/style-images/lee-breathe.jpg', l=256)
style_image = style_image.to(device)
style_image = style_image.repeat(BATCH_SIZE, 1, 1, 1)
_, target_G = loss_network(style_image, input_type = 'style')

# loss function that uses the loss network.
alpha = 1
beta = 1e4
gamma = 1e-5

def train_one_epoch():      
    # Compute the loss and its gradients
    sum_loss = 0.
    sum_L_content = 0.
    sum_L_style = 0.
    sum_L_tv = 0.
    for i, data in enumerate(training_loader):
        y_originals, _ = data
        y_originals = torch.squeeze(y_originals, 1)
        y_originals = y_originals.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        batch_read = len(y_originals)
        y_hats = model(y_originals)
        L_content = 0
        L_style = 0

        # Content and Style Losses
        target_F, _ = loss_network(y_originals)
        F, G = loss_network(y_hats)

        for l in range(len(target_F)):
            L_content += mse_loss(F[l][1], target_F[l][1]) 
        for l in range(len(target_G)):
            L_style += mse_loss(G[l][1], target_G[l][1][:batch_read]) 

        # Total Variation Regularization
        diff_i = torch.sum(torch.abs(y_hats[:, :, :, 1:] - y_hats[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hats[:, :, 1:, :] - y_hats[:, :, :-1, :]))
        L_tv = (diff_i + diff_j)

        loss = alpha*L_content + gamma*L_tv + beta*L_style 

        loss.backward()

        # Adjust learning weights
        sum_L_content += L_content.item()
        sum_L_style += L_style.item()
        sum_L_tv += L_tv.item()
        sum_loss +=loss.item()
        optimizer.step()

        # Gather data and report
        print(f"LOSSES {i+1}/{len(training_loader)}. Total={np.round(sum_loss/(i+1),4)}, Content={np.round(alpha*sum_L_content/(i+1),4)} | Style={np.round(beta*sum_L_style/(i+1),4)} | TV={np.round(gamma*sum_L_tv/(i+1),4)}")

    return sum_loss/len(training_loader)


def main():

    # Initializing in a separate cell so we can easily add more epochs to the same run
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    NUM_EPOCHS = 2500
    print("TRAINING...")
    start = datetime.now()
    for epoch in range(NUM_EPOCHS):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("EPOCH", epoch)
        avg_loss = train_one_epoch()
    print("Elapsed Time:", datetime.now() - start)
    torch.save(model.state_dict(), 'lee-model.pt')
    # torch.save(optimizer.state_dict(), 'lee-model-4-adam.pt')
    
if __name__ == "__main__":
    main()
