import time
import os
import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

##################
# dataset        #
##################

def get_cifar10(args):
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    normalize = transforms.Normalize(mean=means, std=stds)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_folder, train=True, download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_folder, train=False, download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    return trainloader, testloader

def get_imagenet(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


##################
# LR schedules   #
##################

def cosine_lr(cur_epoch, base_lr, max_epoch):
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * cur_epoch / max_epoch))

def adjust_lr(optimizer, cur_epoch, max_epoch, base_lr, warmup_epochs=5, warmup_factor=0.1):
    lr = cosine_lr(cur_epoch, base_lr, max_epoch)

    # Linear warmup
    if cur_epoch < warmup_epochs:
        alpha = cur_epoch / warmup_epochs
        warmup_factor = warmup_factor * (1.0 - alpha) + alpha
        lr *= warmup_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

############
# Training #
############

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=1)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose:
            progress.display(i)

    return losses.avg


def validate(val_loader, model, criterion, args, pct=1.0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    eval_samples = round(pct * len(val_loader.dataset.targets))
    curr_samples = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            curr_samples += len(target)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=1)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose:
                progress.display(i)

            if curr_samples >= eval_samples:
                break

    return losses.avg, top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=1):
    '''Computes the accuracy over the k top predictions'''
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()

