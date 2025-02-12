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


    plt.plot(np.arange(num_epochs), losses_per_epoch)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

    return losses_per_epoch

def image_augmentation():
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=(0, 45)),
                                        transforms.RandomAutocontrast(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Resize((64, 64))])

    dataset_train = ImageFolder('../DATA/clouds/clouds_train', transform=train_transforms)
    dataloader_train = data.DataLoader(dataset_train, shuffle=True, batch_size=1)

    image, label = next(iter(dataloader_train))
    image = image.squeeze().permute(1, 2, 0)
    plt.title(label)
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    image_augmentation()

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=(0, 45)),
                                           transforms.RandomAutocontrast(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Resize((64, 64))])

    dataset_train = ImageFolder('../DATA/clouds/clouds_train', transform=train_transforms)
    dataloader_train = data.DataLoader(dataset_train, shuffle=True, batch_size=16)

    cnn = ConvolutionalNN()
    num_epochs = 20
    learning_rate = 0.001
    optimizer = optim.AdamW(cnn.parameters(), learning_rate)

    loss = train_model(dataloader_train, optimizer, cnn, num_epochs)
    print(f'Average training loss per epoch: {np.mean(loss)}')


if __name__ == '__main__':
    main()