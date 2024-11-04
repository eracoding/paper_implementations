import torch
import torch.nn as nn
import torch.utils.data import DataLoader

from torchvision import transforms, datasets

from vit import ViT
from train import *


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViT((1, 28, 28), n_patches=7, hidden_d=20, n_heads=2, out_d=10)
    model = model.to(device)

    N_EPOCHS = 1
    LR = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()


    train_vit(model, device, train_loader, optimizer, criterion, N_EPOCHS)

    test_vit(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
