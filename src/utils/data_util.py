import torchvision.transforms as transforms


def convert2type_list(str_var, delimiter, var_type):
    return list(map(var_type, str_var.split(delimiter)))


def convert2type_range(str_var, delimiter, var_type):
    return range(*convert2type_list(str_var, delimiter, var_type))


def build_normalizer(dataset, mean=None, std=None):
    if mean is not None and std is not None:
        return transforms.Normalize(mean=mean, std=std)
    return transforms.Normalize(mean=dataset.mean(axis=(0, 1, 2)) / 255, std=dataset.std(axis=(0, 1, 2)) / 255)
