import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data

def main():
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    ])

    dataset_train = ImageFolder(
    '../DATA/clouds/clouds_train',
    transform=train_transforms,
    )

    dataloader_train = data.DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1,
    )


    f, axes = plt.subplots(2, 3, figsize=(10, 6))
    f.suptitle("The clouds dataset")

    for i in range(2):
        for j in range(3):
            image, label = next(iter(dataloader_train))
            image = image.squeeze().permute(1, 2, 0)
            axes[i][j].imshow(image)
            axes[i][j].axis('off')
            axes[i][j].set_title(dataset_train.classes[label.item()])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()