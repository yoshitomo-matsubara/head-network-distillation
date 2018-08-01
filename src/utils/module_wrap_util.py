import sys
import zlib
import torch.nn as nn


class WrapperModule(nn.Module):
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


def wrap_all_child_modules(model, wrapper_module, member_name=None, member_module=None):
    named_children = model.named_children() if member_module is None else member_module.named_children()
    named_children = list(named_children)
    if not named_children and member_name is not None and member_module is not None:
        setattr(model, member_name, wrapper_module(member_module))
        return

    parent = model if member_module is None else member_module
    for name, child_module in named_children:
        wrap_all_child_modules(parent, wrapper_module, name, child_module)
