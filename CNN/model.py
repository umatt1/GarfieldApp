import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, n_input_features):
        super(model, self).__init__()

        # CNN stuff here
        # pooling stuff here
        # CNN stuff here
        # pooling stuff here
        self.linear1 = nn.Linear() # needs input size, hidden size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(,1) # needs in features
        self.sigmoid = nn.Sigmoid()
        return 

    def forward(self, X):
        return 