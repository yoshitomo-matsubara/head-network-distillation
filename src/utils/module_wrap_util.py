import torch.nn as nn


def wrap_all_child_modules(model, wrapper_module, member_name=None, member_module=None):
    named_children = model.named_children() if member_module is None else member_module.named_children()
    named_children = list(named_children)
    if not named_children and member_name is not None and member_module is not None:
        setattr(model, member_name, wrapper_module(member_module))
        return

    for name, child_module in named_children:
        wrap_all_child_modules(model, wrapper_module, name, child_module)


class WrapperModule(nn.Module):
    def __init__(self, org_module):
        super().__init__()
        self.org_module = org_module

    def forward(self, *input):
        output = self.org_module(*input)
        return output
