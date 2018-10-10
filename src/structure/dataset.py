from io import BytesIO

import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as functional
from PIL import Image


class RgbImageDataset(data.Dataset):
    def __init__(self, file_path_lists, size, transform=None, jpeg_quality=0):
        self.transform = transform
        self.jpeg_quality = jpeg_quality
        self.size = size
        self.file_paths = []
        self.labels = []
        self.compression_rates = []
        self.avg_compression_rate = 0
        self.sd_compression_rate = 0
        for class_label, file_path_list in enumerate(file_path_lists):
            for file_path in file_path_list:
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    continue
                self.file_paths.append(file_path)
                self.labels.append(class_label)

    def __len__(self):
        return len(self.labels)

    def compress_img(self, img):
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=95)
        org_file_size = img_buffer.tell()
        img_buffer.close()
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
        comp_file_size = img_buffer.tell()
        recon_img = Image.open(img_buffer)
        return recon_img, comp_file_size / org_file_size

    def __getitem__(self, idx):
        file_path, target = self.file_paths[idx], self.labels[idx]
        img = Image.open(file_path)
        img = functional.resize(img, self.size, interpolation=2)
        if 1 <= self.jpeg_quality <= 95:
            img, compression_rate = self.compress_img(img)
            self.compression_rates.append(compression_rate)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def load_all_data(self):
        data = []
        self.compression_rates = []
        for i in range(len(self.labels)):
            img, _ = self.__getitem__(i)
            data.append(img)

        data = np.concatenate(data)
        if len(self.compression_rates) > 0:
            self.avg_compression_rate = np.average(self.compression_rates)
            self.sd_compression_rate = np.std(self.compression_rates)
            print('Compression rate:', self.avg_compression_rate, '+-', self.sd_compression_rate)
        return data.reshape(len(self.labels), self.size[0], self.size[1], 3)

    def compute_compression_rate(self):
        self.compression_rates = []
        for i in range(len(self.labels)):
            _, _ = self.__getitem__(i)
        self.avg_compression_rate = np.average(self.compression_rates)
        self.sd_compression_rate = np.std(self.compression_rates)
        print('Compression rate:', self.avg_compression_rate, '+-', self.sd_compression_rate)
