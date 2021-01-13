import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import resnet
import utils

def make_cfg(path):
    if path == 'cfg/resnet18_cifar10.py':
        import cfg.resnet18_cifar10
        return cfg.resnet18_cifar10.cfg
    elif path == 'cfg/resnet18_cifar10_gradprune.py':
        import cfg.resnet18_cifar10_gradprune
        return cfg.resnet18_cifar10_gradprune.cfg
    else:
        raise ValueError('{} not an expected cfg file'.format(path))

if __name__=='__main__':
    if len(sys.argv) != 2:
        print("Expect train_cifar10.py [cfg path]")
        exit(1)

    cfg = make_cfg(sys.argv[1])
    model = cfg.arch()
    data = torch.load('saved_models/cifar10_resnet18_1m.pth')
    model.load_state_dict(data['state_dict'])
    _, val_loader = utils.get_cifar10(cfg)

    batch_time = utils.AverageMeter('Time', ':6.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.ProgressMeter(len(val_loader), [top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    model.cuda()
    eval_samples = len(val_loader.dataset.targets)
    curr_samples = 0
    classified_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            curr_samples += len(target)
            target = target.cuda(non_blocking=True)

            # output = model.get_heads(images)
            # houtput = F.softmax(output, dim=2)
            # a, b, c = houtput.shape
            # houtput = houtput.permute(1, 0, 2).contiguous().view(b, a, c)
            # dists = torch.cdist(houtput, houtput)
            # eq_mask = dists.sum((1, 2)) < 5
            # output = output.sum(0)

            # majority vote
            # output = model.masked_heads(images, 4)
            # N = output.shape[0]
            # output_max = output.max(2)[1]
            # eq_tensor = output_max[0].repeat(N).view(N, -1)
            # eq_mask = torch.eq(output_max, eq_tensor).sum(0) >= round(N*0.0)
            # output = output.median(0)[0]

            # entropy
            output = model(images)
            output = F.softmax(output, dim=1)
            ent = (-output*torch.log10(output)).sum(1)
            eq_mask = ent < 0.3

            # measure accuracy and record loss
            acc1 = utils.accuracy(output[eq_mask], target[eq_mask], topk=1)
            top1.update(acc1, images[eq_mask].size(0))
            classified_samples += eq_mask.sum()

            progress.display(i)
        
    print(classified_samples, eval_samples)
