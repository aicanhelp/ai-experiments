import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from dataclasses import dataclass
from torch.multiprocessing import set_start_method
from aiharness.configuration import ArgType, Arguments

try:
    # set for multiprocessing
    set_start_method('spawn')
except RuntimeError:
    pass


@dataclass
class Config:
    batch_size: ArgType = ArgType(32, 'Batch Size for training and testing')
    workers: ArgType = ArgType(8, 'Number of worker threads for dataloading')
    num_epochs: ArgType = ArgType(2, 'Number of epochs to train for')
    starting_lr: ArgType = ArgType(0.1, 'Starting Learning Rate')
    world_size: ArgType = ArgType(4, 'Number of distributed processes')
    dist_backend: ArgType = ArgType('nccl', 'Distributed backend type')
    dist_url: ArgType = ArgType("tcp://172.31.22.234:23456", 'Url used to setup distributed training')
    local_rank: ArgType = ArgType(0, '')
    gpus: ArgType = ArgType(0, '')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class Metrics:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


class DistributedModel:
    def __int__(self, config: Config):
        print("Initialize Model...")
        self.model = models.resnet18(pretrained=False).cuda()
        # device_ids will include all GPU devices by default
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=dp_device_ids,
                                                               output_device=local_rank)
