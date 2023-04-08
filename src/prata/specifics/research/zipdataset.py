import os
from zipfile import ZipFile
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np
from PIL import Image


class _ImageZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_file, samples):
        self.zip_file = zip_file
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        with self.zip_file.open(path) as f:
            img0 = np.array(Image.open(f), dtype=np.uint8)[:, :, ::-1]  # RGB -> BGR

        return path, img0

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str


class ImageZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_path):
        if not os.path.exists(zip_path):
            raise RuntimeError("%s does not exist" % zip_path)

        self.zip_path = zip_path
        self.zipfile = ZipFile(self.zip_path, "r")
        files = self.zipfile.namelist()
        self.samples = []
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                continue
            self.samples.append((file, 0))

    @contextmanager
    def dataset(self):
        res = _ImageZipDataset(
            zip_file=self.zipfile,
            samples=self.samples,
        )
        yield res

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Zip Location: {}\n".format(self.zip_path)
        return fmt_str
