import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_mean_and_std(dataset):
    '''compute the mean and std value of dataset'''
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean    = torch.zeros(3)
    var     = torch.zeros(3)
    print('==> Computing mean and std..')
    imageIdx = 0
    for inputs, targets in dataLoader:
        imageIdx += 1
        if (imageIdx) % 1000 == 0:
            print('starting image no. %d' % imageIdx)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            var[i]  += inputs[:,i,:,:].var()
    mean.div_(len(dataset))
    var.div_((len(dataset))**2)
    std = var
    return mean, std

# Hyper params:
batchSize = 16

trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, shuffle = True)

it = iter(trainLoader)
images, _ = it.next()

torchvision.utils.save_image(images,'grid.png', nrow=4)

mean, std = get_mean_and_std(trainSet)

print(mean)
print(std)