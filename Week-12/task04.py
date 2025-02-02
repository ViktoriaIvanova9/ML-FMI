import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import F1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init

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
        self.fc1 = nn.Linear(9, 32)
        self.bn1 = nn.BatchNorm1d(32)
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')

        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')

        self.fc4 = nn.Linear(8, 4)
        self.bn4 = nn.BatchNorm1d(4)
        init.kaiming_uniform_(self.fc4.weight, nonlinearity='leaky_relu')

        self.fc5 = nn.Linear(4, 2)
        self.bn5 = nn.BatchNorm1d(2)
        init.kaiming_uniform_(self.fc5.weight, nonlinearity='leaky_relu')

        self.fc6 = nn.Linear(2, 1)
        init.kaiming_uniform_(self.fc6.weight, nonlinearity='leaky_relu')

        self.double()

    def forward(self, x):
        x = self.bn1(F.elu(self.fc1(x)))
        x = self.bn2(F.elu(self.fc2(x)))
        x = self.bn3(F.elu(self.fc3(x)))
        x = self.bn4(F.elu(self.fc4(x)))
        x = self.bn5(F.elu(self.fc5(x)))
        x = F.sigmoid(self.fc6(x))
        
        return x
    
def train_model(dataloader_train : DataLoader, optimizer, net, num_epochs): # average loss per epoch
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

    return losses_per_epoch

def main():
    net = Net()
    train_file = '../DATA/water_train.csv'
    test_file = '../DATA/water_test.csv'
    water_potability = WaterDataset(train_file)
    train_dataloader = DataLoader(water_potability, batch_size=64, shuffle=True, drop_last=True)

    test_water_ds = WaterDataset(test_file)
    test_dataloader = DataLoader(test_water_ds, batch_size=64, shuffle=True, drop_last=True)

    num_epochs = 10
    learning_rate = 0.01

    optimizers = {
        'SGD' : optim.SGD(net.parameters(), learning_rate),
        'RMSprop' : optim.RMSprop(net.parameters(), learning_rate),
        'Adam' : optim.Adam(net.parameters(), learning_rate),
        'AdamW' : optim.AdamW(net.parameters(), learning_rate)
    }

    for label, curr_optimizer in optimizers.items():
        print(f'Using the {label} optimizer:')
        loss = train_model(train_dataloader, curr_optimizer, net, num_epochs)
        print(f'Average loss: {np.mean(loss)}')

    loss = train_model(test_dataloader, optim.Adam(net.parameters(), learning_rate), net, num_epochs=1000)
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