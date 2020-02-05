import sys
import time
import zlib

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
