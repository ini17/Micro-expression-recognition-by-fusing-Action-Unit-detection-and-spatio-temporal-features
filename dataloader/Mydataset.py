import random
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import re


AU_CODE = [1, 2, 4, 10, 12, 14, 15, 17, 25]
AU_DICT = {
    number: idx
    for idx, number in enumerate(AU_CODE)
}


class MEDataset(Dataset):
    AMP_LIST = [1.2, 1.4, 1.6, 1.8, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0]

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, catego: str,
                 train: bool, parallel=None, mat_dir=None, au_dir=None):

        self.data_info = data_info
        self.label_mapping = label_mapping
        self.catego = catego
        self.train = train
        self.parallel = parallel
        self.mat_dir = mat_dir
        self.au_dir = au_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        # Label for the image
        label = self.label_mapping[self.data_info.loc[idx, "Estimated Emotion"]]
        # Label for the image
        au_anno = self.data_info.loc[idx, "Action Units"]
        AUs = re.findall(r"\d+", au_anno)
        au_label = [0 for _ in range(9)]
        for au in AUs:
            if au in AU_CODE:
                au_label[AU_DICT[str(au)]] += 1

        subject = self.data_info.loc[idx, "Subject"]
        folder = self.data_info.loc[idx, "Filename"]

        frames = torch.load(f"{self.au_dir}\\sub{subject.zfill(2)}_{folder}.pt")

        n_patches = torch.empty(9, 2, 30, 7, 7)
        for i in range(9):
            if self.train:
                amp_factor = random.choice(MEDataset.AMP_LIST)
            else:
                amp_factor = 2.0
                # onset to apex
                # mat_dir = f"{self.mat_dir}\\Inter_{self.parallel}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
                # onset to offset
            mat_dir = f"{self.mat_dir}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
            for direct in range(2):
                n_patches[i][direct] = torch.load(mat_dir)[i][direct]

        return n_patches, label, frames, torch.Tensor(au_label)
