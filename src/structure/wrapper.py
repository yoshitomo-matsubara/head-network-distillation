import sys
import time
import zlib

import numpy as np
from sklearn.manifold import TSNE
from torch import nn


class CompressionWrapper(nn.Module):
    def __init__(self, org_module, compression_level=9):
        super().__init__()
        self.org_module = org_module
        self.compression_level = compression_level
        self.org_data_size = 0
        self.compressed_data_size = 0
        self.count = 0

    def forward(self, *input):
        output = self.org_module(*input)
        np_output = output.clone().cpu().detach().numpy()
        compressed_output = zlib.compress(np_output, self.compression_level)
        self.org_data_size += np_output.nbytes
        self.compressed_data_size += sys.getsizeof(compressed_output)
        self.count += len(np_output)
        return output

    def get_compression_rate(self):
        return self.compressed_data_size / self.org_data_size

    def get_average_org_data_size(self):
        return self.org_data_size / self.count

    def get_average_compressed_data_size(self):
        return self.compressed_data_size / self.count


class RunTimeWrapper(CompressionWrapper):
    def __init__(self, org_module, compression_level=9):
        super().__init__(org_module, compression_level)
        self.is_first = False
        self.is_compressed = False
        self.start_timestamp_list = list()
        self.timestamp_list = list()
        self.comp_timestamp_list = list()

    def forward(self, *input):
        if self.is_first:
            self.start_timestamp_list.append(time.time())

        output = self.org_module(*input)
        self.timestamp_list.append(time.time())
        if not self.is_compressed:
            return output

        np_output = output.clone().cpu().detach().numpy()
        compressed_output = zlib.compress(np_output, self.compression_level)
        self.org_data_size += np_output.nbytes
        self.compressed_data_size += sys.getsizeof(compressed_output)
        self.count += len(np_output)
        self.comp_timestamp_list.append(time.time())
        return output

    def get_timestamps(self):
        return self.timestamp_list

    def get_compression_timestamps(self):
        return self.comp_timestamp_list

    def get_compression_time_list(self):
        return [self.comp_timestamp_list[i] - self.timestamp_list[i] for i in range(len(self.comp_timestamp_list))]


class RepresentationWrapper(nn.Module):
    def __init__(self, org_module, method='tsne', dim=2):
        super().__init__()
        self.org_module = org_module
        self.method = method
        self.dim = dim
        self.transformed_list = list()

    @staticmethod
    def normalize(np_mat):
        min_values = np.min(np_mat, axis=0, keepdims=True)
        max_values = np.max(np_mat, axis=0, keepdims=True)
        return (np_mat - min_values) / (max_values - min_values)

    def transform_by_tsne(self, np_flat_output):
        transformed_output = TSNE(n_components=self.dim).fit_transform(np_flat_output)
        return self.normalize(transformed_output)

    def forward(self, *input):
        output = self.org_module(*input)
        np_flat_output = output.clone().cpu().detach().flatten(1).numpy()
        if self.method == 'tsne':
            transformed_output = self.transform_by_tsne(np_flat_output)
        else:
            transformed_output = self.normalize(np_flat_output)

        self.transformed_list.append(transformed_output)
        return output

    def get_transformed_list(self):
        return self.transformed_list.copy()
