from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np

class GarfieldDataset(Dataset):

    def __init__(self, dataset_homedir, transform, split=0, debug=False):
        # its easier just to write this ourselves
        # class needs:
        #       open data.csv
        #       load lines (is it a garfield, img, split)
        # dataloader has specified which split it is
        csv = os.path.join(dataset_homedir, "data.csv")
        df = pd.read_csv(csv)
        df = df[df["split"] == split]
        df.reset_index(drop=True, inplace=True)
        self.labels = df["isgarf"]
        self.files = df["filename"]
        self.transform = transform
        self.homedir = dataset_homedir
        return

    def __getitem__(self, index):
        # returns an item
        if self.labels[index] == 1:
            # is a garfield
            file = os.path.join(self.homedir, "garf_pics", self.files[index])
        else:
            file = os.path.join(self.homedir, "non_garf_pics", self.files[index])
        image = Image.open(file).convert('RGB')
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        # len dataset
        return len(self.files)

