import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


'''
a = range(2*3*4)
print(a)
a = torch.Tensor(a).view(3,2,2,2)
print(a)
'''
class OurSimpleModel(nn.Module):
    def __init__(self):
        super(OurSimpleModel, self).__init__()
        self.linear1 = nn.Linear(5,3)
        self.linear2 = nn.Linear(3,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        output = 2*x
        return output

class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.linear1 = nn.Linear(5,3)
        self.linear2 = nn.Linear(3,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.sigmoid(self.linear1(x))
        output = self.sigmoid(self.linear2(x))
        return output

'''
a = Variable(torch.Tensor(range(6*5)).view(6,5))
print "a: ", a.data
'''

x = Variable(torch.Tensor(range(6*5)).view(6,5))
our_model = OurModel()
#print "input to model: ", x
#print "our_model output: ", our_model(x)




def imshow(img,label):
    img = img *255     # unnormalize
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow((np.reshape(npimg, (28,28))).astype(np.uint8), cmap='Greys')
    str = 'label: %d' % label
    fig.suptitle(str)
    return plt

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

image, label = train_dataset[13]

print 'mnist size', len(train_dataset)
#print 'train dataset', train_dataset[60000-1]
print 'single image size: ', image.size()
print 'label: ', label

# show images
plt.close()
imshow(image,label).show()
#imshow(torchvision.utils.make_grid(image))

print 'start train loader'
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

data_iter = iter(train_loader)

images, labels = data_iter.next()
print images.size()
imshow(images[0],labels[0]).show()

