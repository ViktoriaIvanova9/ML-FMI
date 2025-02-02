import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
torch.manual_seed(42)

class WaterDataset(Dataset):
    def __init__(self, filename):
        super(WaterDataset, self).__init__()
        self.water_ds = np.array(pd.read_csv(filename))

    def __len__(self):
        return len(self.water_ds)

    def __getitem__(self, position):
        return (self.water_ds[position][:-1], self.water_ds[position][-1])


class Net(nn.Module):
    def __init__(self): # defines the dimensions of the layers
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16) 
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    
def train_model(dataloader_train : DataLoader, optimizer, net, num_epochs, create_plot=False): # average loss per epoch
    criterion = nn.BCELoss()
    losses_per_epoch = []

    for epoch in tqdm(range(num_epochs)):
        loss_per_feature = []
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            loss_per_feature.append(loss.detach().numpy())

        losses_per_epoch.append(np.mean(loss_per_feature))

    if create_plot:
        plt.plot(np.arange(num_epochs), losses_per_epoch)
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

    return losses_per_epoch

def main():
    net = Net()
    train_file = '../DATA/water_train.csv'
    test_file = '../DATA/water_test.csv'
    water_potability = WaterDataset(train_file)
    train_dataloader = DataLoader(water_potability, batch_size=2, shuffle=True)

    test_water_ds = WaterDataset(test_file)
    test_dataloader = DataLoader(test_water_ds, batch_size=2, shuffle=True)

    num_epochs = 10
    learning_rate = 0.001

    optimizers = {
        'SGD' : optim.SGD(net.parameters(), learning_rate),
        'RMSprop' : optim.RMSprop(net.parameters(), learning_rate),
        'Adam' : optim.Adam(net.parameters(), learning_rate),
        'AdamW' : optim.AdamW(net.parameters(), learning_rate)
    }

    for label, curr_optimizer in optimizers.items():
        print(f'Using the {label} optimizer:')
        loss = train_model(train_dataloader, curr_optimizer, net, num_epochs, create_plot=False)
        print(f'Average loss: {np.mean(loss)}')

    loss = train_model(test_dataloader, optim.AdamW(net.parameters(), learning_rate), net, num_epochs=1000, create_plot=True)
    print(f'Average loss: {np.mean(loss)}')

    f1score = F1Score(task='binary')

    net.eval()
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = net(features)
            preds = (outputs >= 0.5).float()
            f1score(preds, labels.view(-1, 1))

    f1 = f1score.compute()
    print(f'F1 score on test set: {f1}')

if __name__ == '__main__':
    main()
    # Does the model perform well? - Since I received F1 score 0.62, I think no, it should be closer to 1
