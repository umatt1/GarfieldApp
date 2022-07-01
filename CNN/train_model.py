from model import model
import torch

def main():
    # 1. Design the model (look @ model.py)
    network = model
    # 2. Construct loss and optimizer (import from torch)
    learning_rate = 0.01
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 3. Construct and run training loop:
    #   -> forward pass: compute prediction + loss
    #   -> backward pass: gradients
    #   -> update weights
    num_epochs = 100
    for epoch in range(num_epochs):
        y_pred = model.forward()
        loss = criterion()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch+1) % 10 == 0:
            print(f'epoch {epoch+1} loss {loss.item():.4f}')
    print("Hello world!")
    return

if __name__ == "__main__":
    main()