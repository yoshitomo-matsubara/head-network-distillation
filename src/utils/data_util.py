import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image


class RgbImageDataset(data.Dataset):
    def __init__(self, file_path_lists, size, transform=None):
        self.transform = transform
        self.size = size
        self.file_paths = []
        self.labels = []
        for class_label, file_path_list in enumerate(file_path_lists):
            for file_path in file_path_list:
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    continue
                self.file_paths.append(file_path)
                self.labels.append(class_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path, target = self.file_paths[idx], self.labels[idx]
        img = Image.open(file_path)
        img = functional.resize(img, self.size, interpolation=2)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def load_all_data(self):
        data = []
        for i in range(len(self.labels)):
            img, _ = self.__getitem__(i)
            data.append(img)
        data = np.concatenate(data)
        return data.reshape(len(self.labels), self.size[0], self.size[1], 3)


def convert2type_list(str_var, delimiter, var_type):
    return list(map(var_type, str_var.split(delimiter)))


def convert2type_range(str_var, delimiter, var_type):
    return range(*convert2type_list(str_var, delimiter, var_type))


def build_normalizer(dataset):
    mean = dataset.mean(axis=(0, 1, 2)) / 255
    std = dataset.std(axis=(0, 1, 2)) / 255
    return transforms.Normalize(mean=mean, std=std)
