import datetime
import torch
from transfer_vgg_model import NTVGG19
from image_transform_net import ImageTransformer
import pickle
from torch.optim import Adam
from preprocess import load_image_as_tensor
from tqdm import tqdm
from hyperparameters import EPOCHS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('data.pkl', 'rb') as f:
    training_loader = pickle.load(f)

model = ImageTransformer()
model.to(device)

optimizer = Adam(model.parameters(), lr = 0.001)

loss_network = NTVGG19()
loss_network.to(device)

style_image, _ = load_image_as_tensor('sample-van-gogh.jpg')
_, target_G = loss_network(style_image, input_type = 'style')


# loss function that uses the loss network.
alpha = 1
beta = 1e9
def loss_function(y_hat, y_original):
    # print(y_hat.shape)
    # print(y_original)
    L_content = 0
    L_style = 0
    for i in range(len(y_hat)):

        target_F, _ = loss_network(torch.unsqueeze(y_original[i], 0) )

        F, G = loss_network(torch.unsqueeze(y_hat[i], 0))



        for l in range(len(target_F)):
            CHW = target_F[l][1].shape[1] * target_F[l][1].shape[2] * target_F[l][1].shape[3]
            L_content += (torch.norm(F[l][1] - target_F[l][1], p = 2) ** 2) / CHW

        for l in range(len(target_G)):
            # CHW = target_F[l][1].shape[1] * target_F[l][1].shape[2] * target_F[l][1].shape[3]
            # print(G[l][1].shape)
            L_style += (torch.norm(G[l][1] - target_G[l][1], p = 'fro') ** 2) 
    
    loss = alpha*L_content + beta*L_style
    return loss



def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    num_batches = len(training_loader)  

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    batch_num = 1
    for i, data in tqdm(enumerate(training_loader), desc = f'Batch {batch_num}/{num_batches}'):
        
        y_originals, _ = data
        y_originals = torch.squeeze(y_originals)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_hats = model(y_originals)

        # Compute the loss and its gradients
        loss = loss_function(y_hats, y_originals)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        batch_num += 1

        print('batch {} loss: {}'.format(i + 1, loss.item()))

    print('Average loss for epoch: {}'.format(running_loss/num_batches))
    return running_loss/num_batches


def main():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # # Disable gradient computation and reduce memory consumption.
        # with torch.no_grad():
        #     for i, vdata in enumerate(validation_loader):
        #         vinputs, vlabels = vdata
        #         voutputs = model(vinputs)
        #         vloss = loss_fn(voutputs, vlabels)
        #         running_vloss += vloss

        print(f'EPOCH {epoch}, loss: {avg_loss}')

        # # Log the running loss averaged per batch
        # # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #     torch.save(model.state_dict(), model_path)

        epoch_number += 1
    torch.save(model.state_dict(), 'model.pt')
main()