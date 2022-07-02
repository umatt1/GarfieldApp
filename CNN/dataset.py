from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image

class GarfieldDataset(Dataset):

    def __init__(self, data_dir, transform, data_type="train"):
        # Get image file names
        cdm_data = os.path.join(data_dir, data_type)
        file_names = [os.listdir(cdm_data, f) for f in file_names]

        # Get labels
        labels_data = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True)
        # obtained labels from df
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names]
        self.transform = transform
        return

    def __getitem__(self, index):
        # returns an item
        image = Image.open(self.full_filienames[index])
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        # len dataset
        return len(self.full_filenames)

