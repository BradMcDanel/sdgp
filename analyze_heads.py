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

    # batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    model.cuda()
    eval_samples = len(val_loader.dataset.targets)
    curr_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            curr_samples += len(target)
            target = target.cuda(non_blocking=True)

            output = model.masked_heads(images, 16)
            output = F.softmax(output, dim=2).cpu().numpy()
            print(output.shape)
            for j in range(20, 30):
                for k in range(output.shape[0]):
                    plt.plot(output[k, j])
                print(target[j])
                plt.show()

            assert False