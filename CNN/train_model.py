from model import model
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    batch_size=16
    epochs = 10
    # data
    train_dataset = torchvision.datasets.FashionMNIST(root="../../", train=True)
    test_dataset = torchvision.datasets.FashionMNIST(root="../../", train=False)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    examples = iter(train_dataloader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)

    for i in range(6):
        plt.subplot(2,3, i+1)
        plt.imshow(samples[i][0], cmap='gray')
    plt.show()

    # 1. Design the model (look @ model.py)
    network = model((28,28))
    # 2. Construct loss and optimizer (import from torch)
    learning_rate = 0.01
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    # 3. Construct and run training loop:
    #   -> forward pass: compute prediction + loss
    #   -> backward pass: gradients
    #   -> update weights
    n_total_steps = len(train_dataloader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.reshape(-1, (28,28))
            Y_pred = network.forward(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'epoch {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    # test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_dataloader:
            images = images.reshape(-1, (28,28))
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            n_samples += shape[0]
            n_correct += (predictions == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
    return

if __name__ == "__main__":
    main()