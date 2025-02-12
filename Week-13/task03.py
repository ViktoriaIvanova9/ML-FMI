import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import time
import pprint
from torchmetrics import F1Score, Precision, Recall

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flattener = nn.Flatten()
        self.classifier = nn.Linear(64 * 16 * 16, 7)

        self.float()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flattener(x)
        x = self.classifier(x)

        return x
    
def train_model(dataloader, optimizer, net, num_epochs):
    criterion = nn.CrossEntropyLoss()
    losses_per_epoch = []

    for epoch in tqdm(range(num_epochs)):
        loss_per_feature = []
        print(f' Epoch {epoch + 1}', end=" ")
        for images, labels in dataloader:
            images, labels = images.squeeze(), labels.squeeze()
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_per_feature.append(loss.detach().numpy())

        losses_per_epoch.append(np.mean(loss_per_feature))
        print(f'Average training loss per batch: {np.mean(loss_per_feature)}')

    return losses_per_epoch

def compute_precision(cnn, test_dataloader):
    precision_res = Precision(task="multiclass", num_classes=7, average=None)

    cnn.eval()
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = cnn(features)
            preds = torch.argmax(outputs, dim=1)
            precision_res(preds, labels)

    precision = precision_res.compute()
    return torch.mean(precision)

def compute_recall(cnn, test_dataloader):
    recall_res = Recall(task="multiclass", num_classes=7, average=None)

    cnn.eval()
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = cnn(features)
            preds = torch.argmax(outputs, dim=1)
            recall_res(preds, labels)

    recall = recall_res.compute()
    return torch.mean(recall)

def compute_f1_score(cnn, test_dataloader):
    f1score = F1Score(task="multiclass", num_classes=7, average=None)

    cnn.eval()
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = cnn(features)
            preds = torch.argmax(outputs, dim=1)
            f1score(preds, labels)

    f1 = f1score.compute()
    return f1

def main():
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=(0, 45)),
                                           transforms.RandomAutocontrast(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Resize((64, 64))])
    
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((64, 64))])

    dataset_train = ImageFolder('../DATA/clouds/clouds_train', transform=train_transforms)
    dataloader_train = data.DataLoader(dataset_train, shuffle=True, batch_size=16)

    dataset_test = ImageFolder('../DATA/clouds/clouds_test', transform=test_transforms)
    dataloader_test = data.DataLoader(dataset_test, shuffle=True, batch_size=16)

    cnn = ConvolutionalNN()
    num_epochs = 20
    learning_rate = 0.001
    optimizer = optim.AdamW(cnn.parameters(), learning_rate)

    start_time = time.time()
    loss = train_model(dataloader_train, optimizer, cnn, num_epochs)
    end_time = time.time() 

    precision = compute_precision(cnn, dataloader_test)
    recall = compute_recall(cnn, dataloader_test)
    f1 = compute_f1_score(cnn, dataloader_test)

    print()
    print(f'Summary statistics:')
    print(f'Average training loss per epoch: {np.mean(loss)}')
    print(f'Precision: {np.round(precision.item(), 4)}')
    print(f'Recall: {np.round(recall.item(), 4)}')
    print(f'F1: {np.round(torch.mean(f1).item(), 4)}')
    print(f'Total time taken to train the model in seconds: {end_time - start_time}')

    classes_names = dataset_test.classes
    f1_per_class = {class_name : np.round(f1_score.item(), 4) for class_name, f1_score in zip(classes_names, f1)}

    print()
    print(f'Per class F1 score')
    pprint.pprint(f1_per_class)


if __name__ == '__main__':
    main()

    # I will analyse the result given in the Week-13 task:
    # stratiform clouds are classified best
    # cumulonimbus clouds are not guessed well

    # the overal F1score is bad