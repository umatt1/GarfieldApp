import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, n_input_features):
        super(model, self).__init__()

        # CNN stuff here
        self.conv1 = nn.Conv2d(n_input_features, 128, (5,5), stride=1, padding='same')
        # pooling stuff here
        self.pool = nn.MaxPool2d((2,2))
        # CNN stuff here
        self.conv2 = nn.Conv2d(128, 128, (5,5), stride=1, padding='same')
        # pooling stuff here
        self.linear1 = nn.Linear(6272,640) # needs input size, hidden size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(640,2) # needs in features
        self.sigmoid = nn.Sigmoid()
        return 

    def forward(self, X):
        X = self.conv1(X)
        X = self.pool(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.pool(X)
        X = self.relu(X)
        X = X.view(-1, 128*7*7)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.sigmoid(X)
        return X