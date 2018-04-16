import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

def imshow(img,title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    str = '%s' % title
    fig.suptitle(str)
    return plt

def imshowGreyscale(img,title):
    img = img *255     # unnormalize
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow((np.reshape(npimg, (32,32))).astype(np.uint8), cmap='Greys')
    str = '%s' % title
    fig.suptitle(str)
    return plt

# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2434, 0.2615)),])

# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/',train=True,transform=transform,download=True)
test_dataset = dsets.CIFAR10(root='./data/',train=False,transform=transform,download=True)

print('in CIFAR10 train_dataset there are %d images' % train_dataset.train_labels.__len__())
print('in CIFAR10 test_dataset there are %d images' % test_dataset.test_labels.__len__())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(8*8*32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)

class CNN_Greyscale(nn.Module):
    def __init__(self):
        super(CNN_Greyscale, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(8*8*32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

cnn = CNN()
cnnGreyScale = CNN_Greyscale()

if torch.cuda.is_available():
    cnn = cnn.cuda()
    cnnGreyScale = cnnGreyScale.cuda()

# learn some cnn layers:
# learnConv2d:
class CNN_Conv2d(nn.Module):
    def __init__(self):
        super(CNN_Conv2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5, padding=2))

    def forward(self, x):
        out = self.layer1(x)
        return out


class CNN_Conv2d_BatchNorm2d(nn.Module):
    def __init__(self):
        super(CNN_Conv2d_BatchNorm2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16))

    def forward(self, x):
        out = self.layer1(x)
        return out

class CNN_Conv2d_BatchNorm2d_ReLU(nn.Module):
    def __init__(self):
        super(CNN_Conv2d_BatchNorm2d_ReLU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        return out

class CNN_Conv2d_BatchNorm2d_ReLU_MaxPool2d(nn.Module):
    def __init__(self):
        super(CNN_Conv2d_BatchNorm2d_ReLU_MaxPool2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        out = self.layer1(x)
        return out



cnnConv2d = CNN_Conv2d()
cnnConv2dBatchNorm2d = CNN_Conv2d_BatchNorm2d()
cnnConv2dBatchNorm2dReLU = CNN_Conv2d_BatchNorm2d_ReLU()
cnnConv2dBatchNorm2dReLUMaxPool2d = CNN_Conv2d_BatchNorm2d_ReLU_MaxPool2d()

train_loader_singleImage = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=2,shuffle=True)
it = iter(train_loader_singleImage)
imagesLearn,_ = it.next()
imagesLearn = to_var(imagesLearn)

conv2dOut = cnnConv2d(imagesLearn)
conv2dBatchNorm2dOut = cnnConv2dBatchNorm2d(imagesLearn)
conv2dBatchNorm2dReLUOut = cnnConv2dBatchNorm2dReLU(imagesLearn)
conv2dBatchNorm2dReLUMaxPool2dOut = cnnConv2dBatchNorm2dReLUMaxPool2d(imagesLearn)

print('when size of input to conv2d: %dx%dx%dx%d' % (imagesLearn.data.size()[0],imagesLearn.data.size()[1],imagesLearn.data.size()[2],imagesLearn.data.size()[3]))
print('then size of out from conv2d: %dx%dx%dx%d' % (conv2dOut.data.size()[0],conv2dOut.data.size()[1],conv2dOut.data.size()[2],conv2dOut.data.size()[3]))
print 'number of parameters @ conv2d: ', sum(param.numel() for param in cnnConv2d.parameters())
print('then size of out from conv2d->BatchNorm2d: %dx%dx%dx%d' % (conv2dBatchNorm2dOut.data.size()[0],conv2dBatchNorm2dOut.data.size()[1],conv2dBatchNorm2dOut.data.size()[2],conv2dBatchNorm2dOut.data.size()[3]))
print('then size of out from conv2d->BatchNorm2d->ReLU: %dx%dx%dx%d' % (conv2dBatchNorm2dReLUOut.data.size()[0],conv2dBatchNorm2dReLUOut.data.size()[1],conv2dBatchNorm2dReLUOut.data.size()[2],conv2dBatchNorm2dReLUOut.data.size()[3]))
print('then size of out from conv2d->BatchNorm2d->ReLU->MaxPool2d(2): %dx%dx%dx%d' % (conv2dBatchNorm2dReLUMaxPool2dOut.data.size()[0],conv2dBatchNorm2dReLUMaxPool2dOut.data.size()[1],conv2dBatchNorm2dReLUMaxPool2dOut.data.size()[2],conv2dBatchNorm2dReLUMaxPool2dOut.data.size()[3]))
# end of learning


# convert all the weights tensors to cuda()
# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
optimizerGreySclae = torch.optim.Adam(cnnGreyScale.parameters(), lr=learning_rate)
print 'number of parameters: ', sum(param.numel() for param in cnn.parameters())

# Train:
for epoch in range(num_epochs):
    # at each epoch we train the net on all 50000 images, i.e on all the train_dataset.
    # a single optimization step is performed per batchSize images that the net has seen.
    # therefor, the number of optimization steps performed in a single epoch is 50000/batchSize
    for i, (images, labels) in enumerate(train_loader):
        imagesGreyScale = images.sum(dim=1)
        imagesGreyScale = imagesGreyScale[:, np.newaxis, :, :]
        #imshowGreyscale((imagesGreyScale[0] / imagesGreyScale[0].max()),"i: %d" % i).show()

        # at images if shuffle is set to True the order of the images changes
        imagesGreyScale = to_var(imagesGreyScale)
        images = to_var(images)
        labels = to_var(labels)
        # images contains batchSize images, each image is a matrix [3x32x32]

        #imshow(torchvision.utils.make_grid(images.data / images.data.max()), "i: %d" %i).show()



        # Forward + Backward + Optimize
        optimizer.zero_grad()
        optimizerGreySclae.zero_grad()
        outputs = cnn(images)
        outputsGreyScale = cnnGreyScale(imagesGreyScale)
        # outputs contains 100 results, each result is a column vector [10x1]

        loss = criterion(outputs, labels)
        lossGreyScale = criterion(outputsGreyScale, labels)
        loss.backward()
        lossGreyScale.backward()
        optimizer.step()
        optimizerGreySclae.step()
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'%(epoch+1, num_epochs, i+1,len(train_dataset)//batch_size, loss.data[0]))


# Change model to 'eval' mode (BN uses moving mean/var, stopping dropout).
cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = to_var(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print 'Test Accuracy of the model on the 10000 test images: %d %%' % (100 *correct / total)

cnnGreyScale.eval()
correct = 0
total = 0
for images, labels in test_loader:
    imagesGreyScale = images.sum(dim=1)
    imagesGreyScale = to_var(imagesGreyScale)
    outputs = cnnGreyScale(imagesGreyScale)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print 'Test Accuracy of the GreyScaleModel on the 10000 test images: %d %%' % (100 *correct / total)


# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')