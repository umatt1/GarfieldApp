from torch.utils.data import Dataset, DataLoader

class GarfieldDataset(Dataset):
    def __init__(self):
        # data loading
        # pull up the data somehwo on this line
        self.x = # tensor
        self.y = # tensor
        self.n_samples = # shape of a tensor
        return

    def __getitem__(self, index):
        # returns an item
        return self.x[index], self.y[index]
    def __len__(self):
        # len dataset
        return self.n_samples

