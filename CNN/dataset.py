from torch.utils.data import Dataset, DataLoader

class GarfieldDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        # pull up the data somehwo on this line
        self.x = # tensor
        self.y = # tensor
        self.n_samples = # shape of a tensor

        self.transform = transform
        return

    def __getitem__(self, index):
        # returns an item
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        # len dataset
        return self.n_samples

