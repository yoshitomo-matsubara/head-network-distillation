from io import BytesIO

import numpy as np
import torchvision.transforms.functional as functional
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from myutils.pytorch.vision.dataset import RgbImageDataset


class AdvRgbImageDataset(RgbImageDataset):
    def __init__(self, file_path, size, transform=None, jpeg_quality=0):
        super().__init__(file_path, size, transform=transform, delimiter='\t')
        self.jpeg_quality = jpeg_quality
        self.compression_rates = []
        self.avg_compression_rate = 0
        self.sd_compression_rate = 0

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
        if img.mode == 'L':
            img = img.convert('RGB')

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


class AdvImageFolder(ImageFolder):
    def __init__(self, root, size, transform=None, target_transform=None, loader=default_loader, jpeg_quality=0):
        super().__init__(root, transform, target_transform, loader)
        self.size = size
        self.jpeg_quality = jpeg_quality
        self.compression_rates = []
        self.avg_compression_rate = 0
        self.sd_compression_rate = 0

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
        file_path, target = self.samples[idx]
        img = Image.open(file_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = functional.resize(img, self.size, interpolation=2)
        if 1 <= self.jpeg_quality <= 95:
            img, compression_rate = self.compress_img(img)
            self.compression_rates.append(compression_rate)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def load_all_data(self):
        data = []
        self.compression_rates = []
        for i in range(len(self.targets)):
            img, _ = self.__getitem__(i)
            data.append(img)

        data = np.concatenate(data)
        if len(self.compression_rates) > 0:
            self.avg_compression_rate = np.average(self.compression_rates)
            self.sd_compression_rate = np.std(self.compression_rates)
            print('Compression rate:', self.avg_compression_rate, '+-', self.sd_compression_rate)
        return data.reshape(len(self.targets), self.size[0], self.size[1], 3)

    def compute_compression_rate(self):
        self.compression_rates = []
        for i in range(len(self.targets)):
            _, _ = self.__getitem__(i)

        self.avg_compression_rate = np.average(self.compression_rates)
        self.sd_compression_rate = np.std(self.compression_rates)
        print('Compression rate:', self.avg_compression_rate, '+-', self.sd_compression_rate)

