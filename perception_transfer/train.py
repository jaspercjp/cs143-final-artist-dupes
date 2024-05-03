import datetime
import torch
from transfer_vgg_model import NTVGG19
from image_transform_net import ImageTransformer, ImageTransformerRef
import pickle
from torch.optim import Adam
from preprocess import load_image_as_tensor, vgg_normalize, LandscapeDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hyperparameters import EPOCHS
from torch.nn.functional import mse_loss
import torch.nn as nn
from residual_block import ResidualBlock
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 2
relative_root = '../data/'
dataset = LandscapeDataset(relative_root=relative_root)
training_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle = True)

model = ImageTransformerRef().to(device)
optimizer = Adam(model.parameters(), lr = 0.0001)

loss_network = NTVGG19()
loss_network.to(device)

style_image, _ = load_image_as_tensor('../sample-van-gogh.jpg', l=256)
style_image = style_image.to(device)
style_image = style_image.repeat(BATCH_SIZE, 1, 1, 1)
print(style_image.shape)
_, target_G = loss_network(style_image, input_type = 'style')

# loss function that uses the loss network.
alpha = 1
beta = 1e4
gamma = 1e-5
def loss_function(y_hat, y_original, batch_num, total_batch):
    L_content = 0
    L_style = 0
    
    # Content and Style Losses
    target_F, _ = loss_network(y_original)
    F, G = loss_network(y_hat)
     # print("G Shape: ", G[0][1].shape)
    for l in range(len(target_F)):
        L_content += mse_loss(F[l][1], target_F[l][1]) / len(target_F)
    for l in range(len(target_G)):
        L_style += mse_loss(G[l][1], target_G[l][1]) / len(target_G) 
    
    # Total Variation Regularization
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    L_tv = (diff_i + diff_j)
    loss = alpha*L_content + beta*L_style + gamma*L_tv
    print(f"LOSSES {batch_num}/{total_batch}. Content={alpha*L_content.item()} | Style={beta*L_style.item()} | TV={gamma*L_tv.item()}")
    return loss


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    num_batches = len(training_loader)  

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        
        y_originals, _ = data
        y_originals = torch.squeeze(y_originals, 1)
        y_originals = y_originals.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_hats = model(y_originals)
        
        # Compute the loss and its gradients
        loss = loss_function(y_hats, y_originals, i, len(training_loader))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss/num_batches


def main():

    # Initializing in a separate cell so we can easily add more epochs to the same run
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    print("TRAINING...")
    for epoch in range(2):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("EPOCH", epoch)
        avg_loss = train_one_epoch()
    torch.save(model.state_dict(), 'model.pt')
main()
