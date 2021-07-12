from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from gsr import GSRConv2d


def nongrad_param(x):
    return nn.Parameter(x, requires_grad=False)

def make_pair(x):
    if type(x) == int or type(x) == float:
        return x, x
    
    return x

class StochasticPruning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, percentile): 
        percentile = torch.Tensor([percentile])  
        ctx.save_for_backward(percentile)  
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # print((grad_output == 0).sum() / grad_output.nelement())
        # plt.hist(grad_output.view(-1).cpu().tolist(), bins=100)
        # plt.xscale('log')
        # plt.show()
        # return grad_output, None

        percentile, = ctx.saved_tensors
        tau = torch.quantile(grad_output.abs(), percentile.item())
        tau = tau.item()
        r = torch.rand(grad_output.shape, device=grad_output.device)
        ind_a = grad_output.abs() < tau
        ind_b = grad_output.abs() < (tau * r)
        ind_c = grad_output < 0

        grad_output[ind_a & ind_b]  = 0
        grad_output[ind_a & (~ind_b) & ind_c] = -tau
        grad_output[ind_a & (~ind_b) & (~ind_c)] = tau

        # plt.hist(grad_output.view(-1).cpu().tolist(), bins=100)
        # plt.show()
        # no_change = (grad_output.abs() > tau).sum()
        # print('--------')
        # print(tau, percentile)
        # print(no_change / grad_output.nelement())
        # print(grad_output.shape)
        # print('---------')

        return grad_output, None



class Net(nn.Module):
    def __init__(self, quant_params):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = GSRConv2d(quant_params, 32, 64, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = StochasticPruning.apply(x, 0.6)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = StochasticPruning.apply(x, 0.6)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.FashionMNIST('../data', train=True, download=True,
                                     transform=transform)
    dataset2 = datasets.FashionMNIST('../data', train=False,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    quant_params = {
        'w_bits': 3,
        'x_bits': 3,
        'g_bits': 5,
        'g_groupsize': 4,
        'g_nonzero': 1,
    }
    model = Net(quant_params).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()