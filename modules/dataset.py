import os

import torch
from torch.utils.data import Dataset
from skimage import io

from utils import generate_phoc_vector, generate_phos_vector

import pandas as pd
import numpy as np

class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):
        self.df_all = pd.read_csv(csvfile, usecols = ["Image", "Word"])
        self.root_dir = root_dir
        self.transform = transform
        self.calc_phosc = calc_phosc

        words = self.df_all["Word"].values

        phos_vects = []
        phoc_vects = []
        phosc_vects = []

        for word in words:
            phos = generate_phos_vector(word)
            phoc = np.array(generate_phoc_vector(word))
            phosc = np.concatenate((phos, phoc))

            phos_vects.append(phos)
            phoc_vects.append(phoc)
            phosc_vects.append(phosc)

        self.df_all["phos"] = phos_vects
        self.df_all["phoc"] = phoc_vects
        self.df_all["phosc"] = phosc_vects

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = io.imread(img_path)

        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        if self.transform:
            image = self.transform(image)

        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms

    dataset = phosc_dataset('image_data/IAM_test_unseen.csv', '../image_data/IAM_test', transform=transforms.ToTensor())

    print(dataset.df_all)

    print(dataset.__getitem__(0))
