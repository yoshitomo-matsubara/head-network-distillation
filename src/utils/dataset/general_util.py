import multiprocessing

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms

from structure.dataset import AdvRgbImageDataset
from utils import data_util


def get_test_transformer(dataset_name, normalizer, compression_type, compressed_size, org_size):
    normal_list = [transforms.CenterCrop(org_size)] if dataset_name == 'imagenet' else []
    normal_list.append(transforms.ToTensor())
    if normalizer is not None:
        normal_list.append(normalizer)

    normal_transformer = transforms.Compose(normal_list)
    if compression_type is None or compressed_size is None:
        return normal_transformer

    if compression_type == 'base':
        comp_list = [transforms.Resize(compressed_size), transforms.Resize(org_size), transforms.ToTensor()]
        if normalizer is not None:
            comp_list.append(normalizer)
        return transforms.Compose(comp_list)
    return normal_transformer


def get_data_loaders(dataset_config, batch_size=100, compression_type=None, compressed_size=None, normalized=True,
                     rough_size=None, reshape_size=(224, 224), jpeg_quality=0, distributed=False):
    data_config = dataset_config['data']
    dataset_name = dataset_config['name']
    train_file_path = data_config['train']
    valid_file_path = data_config['valid']
    test_file_path = data_config['test']
    normalizer_config = dataset_config['normalizer']
    mean = normalizer_config['mean']
    std = normalizer_config['std']
    train_dataset = AdvRgbImageDataset(train_file_path, reshape_size)
    normalizer = data_util.build_normalizer(train_dataset.load_all_data() if mean is None or std is None else None,
                                            mean, std) if normalized else None
    train_comp_list = [transforms.Resize(rough_size), transforms.RandomCrop(reshape_size)]\
        if rough_size is not None else list()
    train_comp_list.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    valid_comp_list = [transforms.ToTensor()]
    if normalizer is not None:
        train_comp_list.append(normalizer)
        valid_comp_list.append(normalizer)

    pin_memory = torch.cuda.is_available()
    num_cpus = multiprocessing.cpu_count()
    num_workers = data_config.get('num_workers', 0 if num_cpus == 1 else min(num_cpus, 8))
    train_transformer = transforms.Compose(train_comp_list)
    valid_transformer = transforms.Compose(valid_comp_list)
    train_dataset = AdvRgbImageDataset(train_file_path, reshape_size, train_transformer)
    valid_dataset = AdvRgbImageDataset(valid_file_path, reshape_size, valid_transformer)
    test_transformer = get_test_transformer(dataset_name, normalizer, compression_type, compressed_size, reshape_size)
    test_reshape_size = rough_size if dataset_name == 'imagenet' else reshape_size
    test_dataset = AdvRgbImageDataset(test_file_path, test_reshape_size, test_transformer, jpeg_quality)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        test_sampler = SequentialSampler(test_dataset)

    train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, shuffle=True, sampler=train_batch_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                              sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
    if 1 <= test_dataset.jpeg_quality <= 95:
        test_dataset.compute_compression_rate()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
