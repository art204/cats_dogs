import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InferenceDataset(Dataset):

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.all_files = []
        self.all_labels = []
        self.images = []
        self.classes = None

        if os.path.isdir(os.path.join(self.root, os.listdir(self.root)[0])):
            self.classes = sorted(os.listdir(self.root))

        if self.classes is not None:
            for i, class_name in enumerate(self.classes):
                files = sorted(os.listdir(os.path.join(self.root, class_name)))
                self.all_files += files
                self.all_labels += [i] * len(files)
        else:
            self.all_files = sorted(os.listdir(self.root))
            self.all_labels = [np.NaN] * len(self.all_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, item):
        label = self.all_labels[item]
        filename = self.all_files[item]
        if self.classes is not None:
            file_path = os.path.join(self.root, self.classes[label], filename)
        else:
            file_path = os.path.join(self.root, filename)
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, file_path
