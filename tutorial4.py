import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img,title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    str = '%s' % title
    fig.suptitle(str)
    return plt

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=False, transform=transforms.ToTensor())

print('in CIFAR10 trainset there are %d images' % trainset.train_labels.__len__())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)
it = iter(trainloader)
images,_ = it.next()

# show images
imshow(torchvision.utils.make_grid(images),'orig').show()
torchvision.utils.save_image(images,'grid.png',nrow=4)

print trainset.train_data.mean(axis=(0,1,2))/255.0
print trainset.train_data.std(axis=(0,1,2))/255.0
# division by 255.0 because the data is given in bytes.

print('beginning image normalization')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434,0.2615))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)
it = iter(trainloader)
imagesNorm,_ = it.next()

imshow(torchvision.utils.make_grid(imagesNorm/imagesNorm.max()),'norm').show() # /imagesNorm.max() because for show images values must be [0,1]
torchvision.utils.save_image(imagesNorm/imagesNorm.max(),'grid_norm.png',nrow=4)

# Convolution:

im = Variable(imagesNorm)
conv = nn.Conv2d(3,16,5,1)
# 3 input maps, 16 output maps
# 5x5 kernels, 2x2 strides, without padding
output = conv(im)

print conv.weight.size()
torchvision.utils.save_image(im[0].data,'before_conv.png')
torchvision.utils.save_image(output[0].unsqueeze(1).data,'conv_output.png',nrow=4)
imshow(torchvision.utils.make_grid(im[0].data/im[0].data.max()),'before_conv').show()

outFig = output[0].unsqueeze(1).data - output[0].unsqueeze(1).data.min()
outFig = outFig/outFig.max()
imshow(torchvision.utils.make_grid(outFig),'conv_output').show()

x=3