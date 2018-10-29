import sys
import time
import zlib

import torch.nn as nn


class CompressionWrapper(nn.Module):
    def __init__(self, org_module, compression_level=9):
        super().__init__()
        self.org_module = org_module
        self.compression_level = compression_level
        self.org_bandwidth = 0
        self.compressed_bandwidth = 0
        self.count = 0

    def forward(self, *input):
        output = self.org_module(*input)
        np_output = output.clone().cpu().detach().numpy()
        compressed_output = zlib.compress(np_output, self.compression_level)
        self.org_bandwidth += np_output.nbytes
        self.compressed_bandwidth += sys.getsizeof(compressed_output)
        self.count += len(np_output)
        return output

    def get_compression_rate(self):
        return self.compressed_bandwidth / self.org_bandwidth

    def get_average_org_bandwidth(self):
        return self.org_bandwidth / self.count

    def get_average_compressed_bandwidth(self):
        return self.compressed_bandwidth / self.count


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
        self.org_bandwidth += np_output.nbytes
        self.compressed_bandwidth += sys.getsizeof(compressed_output)
        self.count += len(np_output)
        self.comp_timestamp_list.append(time.time())
        return output

    def get_timestamps(self):
        return self.timestamp_list

    def get_compression_timestamps(self):
        return self.comp_timestamp_list

    def get_compression_time_list(self):
        return [self.comp_timestamp_list[i] - self.timestamp_list[i] for i in range(len(self.comp_timestamp_list))]
