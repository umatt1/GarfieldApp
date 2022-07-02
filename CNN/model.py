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
        self.conv2 = nn.Conv2d()
        # pooling stuff here
        self.linear1 = nn.Linear(n_input_features, ) # needs input size, hidden size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(0,1) # needs in features
        self.sigmoid = nn.Sigmoid()
        return 

    def forward(self, X):
        import pdb; pdb.set_trace()
        X = self.conv1(X)
        X = self.pool(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.pool(X)
        X = self.relu(X)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.sigmoid(X)
        return X