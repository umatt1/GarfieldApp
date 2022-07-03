from model import model
import torch
from dataset import GarfieldDataset
import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn as nn

def main():
    FILE = "model.pth"
    batch_size=16
    epochs = 10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(250,250)), 
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            
        ])

    # data
    train_dataset = GarfieldDataset(os.path.join(os.getcwd(), "dataset"), transform, split=0)
    test_dataset = GarfieldDataset(os.path.join(os.getcwd(), "dataset"), transform, split=1)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    examples = iter(train_dataloader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)

    #for i in range(6):
    #    plt.subplot(2,3, i+1)
    #    plt.imshow(samples[i][0], cmap='gray')
    #plt.show()

    # 1. Design the model (look @ model.py)
    network = models.resnet18(pretrained=True)
    # transfer learning
    for param in network.parameters():
        param.requires_grad = False
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 2)
    # 2. Construct loss and optimizer (import from torch)
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # 3. Construct and run training loop:
    #   -> forward pass: compute prediction + loss
    #   -> backward pass: gradients
    #   -> update weights
    n_total_steps = len(train_dataloader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # forward pass, data manipulation
            #images = images.reshape(-1, 28,28)
            Y_pred = network.forward(images)
            loss = criterion(Y_pred, labels)
            #print(Y_pred, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                print(f'epoch {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    # test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_dataloader:
            #images = images.reshape(-1, (28,28))
            outputs = network.forward(images)
            print(outputs)
            print(labels)
            _, predictions = torch.max(outputs, 1)
            n_samples += images.shape[0]
            n_correct += (predictions == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(acc)

    torch.save(network.state_dict(), FILE)
    return

if __name__ == "__main__":
    main()