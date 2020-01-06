import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from structure.dataset import AdvImageFolder
from utils.dataset import imagenet_util


def setup_model(args):
    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    return model


def resume_from_ckpt(model, optimizer, args):
    # optionally resume from a checkpoint
    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return best_prec1


def get_training_data_loader_and_sampler(train_dir, args, normalize):
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_loader, train_sampler


def get_validation_data_loader(valid_dir, args, normalize):
    if args.arch == 'inception_v3':
        rough_size = 299
        valid_transformer = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ])
    else:
        rough_size = 256
        valid_transformer = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return torch.utils.data.DataLoader(
        AdvImageFolder(valid_dir, rough_size, valid_transformer, jpeg_quality=args.jpeg_quality),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )


def train_model(model, train_loader, valid_loader, train_sampler, criterion, optimizer, best_prec1, args):
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        imagenet_util.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        imagenet_util.train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        prec1 = imagenet_util.validate(valid_loader, model, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        imagenet_util.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best)


def main(args):
    model = setup_model(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_prec1 = resume_from_ckpt(model, optimizer, args)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(args.data, 'train')
    valid_dir = os.path.join(args.data, 'val')
    valid_loader = get_validation_data_loader(valid_dir, args, normalize)
    if not args.evaluate:
        train_loader, train_sampler = get_training_data_loader_and_sampler(train_dir, args, normalize)
        train_model(model, train_loader, valid_loader, train_sampler, criterion, optimizer, args)

    imagenet_util.validate(valid_loader, model, criterion, args)
    if args.jpeg_quality > 0 and not args.skip_comp_rate:
        valid_loader.dataset.compute_compression_rate()


if __name__ == '__main__':
    argparser = imagenet_util.get_argparser()
    main(argparser.parse_args())
