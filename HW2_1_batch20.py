import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# Hyper Parameters
num_epochs      = 100
batch_size      = 20
learning_rate   = 0.0005
momentum        = 0.0

# Image Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.2434, 0.2615)),])

# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/',train=True,transform=transform_train,download=True)
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
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
            )
        self.b1 = nn.Sequential(
            nn.Conv2d(48, 12, kernel_size=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True)
            )
        self.b2 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(48, 12, kernel_size=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
        )

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(16*16*56, 10)
        self.fc2 = nn.Linear(20, 10)
        self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        y1 = self.b1(out)
        y2 = self.b2(out)
        y3 = self.b3(out)
        y4 = self.b4(out)
        interceptionOut = torch.cat([y1, y2, y3, y4], 1)
        out = self.maxpool(interceptionOut)
        #out = self.dropout(out)
        out = out.view(out.size(0), -1)

        #out = self.layer3(out)
        #out = out.view(out.size(0), -1)
        #out = self.dropout(out)
        out = self.fc(out)
        #out = self.fc2(out)
        #return self.logsoftmax(out)
        return out

# GoogleNet:(192,  64,  96, 128, 16, 32, 32)
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):

        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 177, kernel_size=3, padding=1),
            nn.BatchNorm2d(177),
            nn.ReLU(True),
        )
        # (1,3,5,6)
        # 16 + 32 + 8 = 64
        self.a3 = Inception(177, 64/4/2, 96/4/2, 128/4/2, 16/4/2, 32/4/2, 32/4/2)  # 165,552 params
        # 32 + 48 + 24 + 16 = 120
        self.b3 = Inception(32, 128/4/2, 128/4/2, 192/4/2, 32/4/2, 96/4/2, 64/4/2)  # 424,092

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 48 + 52 + 12 + 16 = 128
        self.a4 = Inception(60, 192/4/2, 96/4/2, 208/4/2, 16/4/2, 48/4/2, 64/4/2)
        # 160+224+64+64 = 512
        self.b4 = Inception(64, 160/4, 112/4, 224/4, 24/4, 64/4, 64/4)
        '''
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        '''
        self.avgpool = nn.AvgPool2d(16, stride=1)

        # pre_layers: 5760 params
        '''
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)           # 165,552 params            
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)         # 424,092

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        '''
        #self.avgpoolPreLayers = nn.AvgPool2d(16, stride=1)
        
        #self.linear = nn.Linear(1024, 10) # 10250 params
        #self.linear480 = nn.Linear(480, 10)  # 10250 params
        self.linear512 = nn.Linear(512/4, 10)  # 10250 params

    def forward(self, x):
        xSize = x.data.size()

        out = self.pre_layers(x)
        preLayersOutSize = out.data.size()



        out = self.a3(out)
        a3OutSize = out.data.size()

        out = self.b3(out)
        b3OutSize = out.data.size()

        out = self.maxpool(out)
        maxPoolOutSize = out.data.size()

        #out = self.avgpoolPreLayers(out)
        #avgpoolPreLayersSize = out.data.size()


        out = self.a4(out)
        a4OutSize = out.data.size()

        out = self.b4(out)
        b4OutSize = out.data.size()
        '''
        out = self.c4(out)
        c4OutSize = out.data.size()

        out = self.d4(out)
        d4OutSize = out.data.size()

        out = self.e4(out)
        e4OutSize = out.data.size()

        out = self.maxpool(out)
        maxPool2OutSize = out.data.size()

        out = self.a5(out)
        a5OutSize = out.data.size()

        out = self.b5(out)
        b5OutSize = out.data.size()
        '''


        out = self.avgpool(out)
        avgPoolOutSize = out.data.size()

        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #out = self.linear480(out)
        out = self.linear512(out)

        #avgpoolPreLayersOut = avgpoolPreLayersOut.view(out.size(0), -1)
        #currentOut = self.linear(avgpoolPreLayersOut)

        return out#, maxPoolOutSize, avgpoolPreLayersSize #, xSize, preLayersOutSize, a3OutSize, b3OutSize, maxPoolOutSize, a4OutSize, b4OutSize, c4OutSize, d4OutSize, e4OutSize, maxPool2OutSize, a5OutSize, b5OutSize, avgPoolOutSize

# End of GoogleNet

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


#cnn = CNN()
cnn = GoogLeNet()

if torch.cuda.is_available():
    cnn = cnn.cuda()

# convert all the weights tensors to cuda()
# Loss and Optimizer
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(cnn.parameters(), lr = learning_rate, momentum=momentum)
#optimizer = torch.optim.SGD(cnn.parameters(), lr = learning_rate)
print 'number of parameters: ', sum(param.numel() for param in cnn.parameters())

# Train:
for epoch in range(num_epochs):
    # at each epoch we train the net on all 50000 images, i.e on all the train_dataset.
    # a single optimization step is performed per batchSize images that the net has seen.
    # therefor, the number of optimization steps performed in a single epoch is 50000/batchSize
    for i, (images, labels) in enumerate(train_loader):
        # at images if shuffle is set to True the order of the images changes
        images = to_var(images)
        labels = to_var(labels)
        # images contains batchSize images, each image is a matrix [3x32x32]

        #imshow(torchvision.utils.make_grid(images.data / images.data.max()), "i: %d" %i).show()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        #outputs, maxPoolOutSize, avgpoolPreLayersSize  = cnn(images)
        '''
        outputs, xSize, preLayersOutSize, a3OutSize, b3OutSize, maxPoolOutSize, a4OutSize, b4OutSize, c4OutSize, d4OutSize, e4OutSize, maxPool2OutSize, a5OutSize, b5OutSize, avgPoolOutSize = cnn(images)
        if ((epoch==0) and (i==0)):
            print('inputSize: %dx%dx%dx%d' % (xSize[0], xSize[1], xSize[2], xSize[3]))
            print('preLayersOutSize: %dx%dx%dx%d' % (preLayersOutSize[0], preLayersOutSize[1], preLayersOutSize[2], preLayersOutSize[3]))
            print('a3OutSize: %dx%dx%dx%d' % (a3OutSize[0], a3OutSize[1], a3OutSize[2], a3OutSize[3]))
            print('b3OutSize: %dx%dx%dx%d' % (b3OutSize[0], b3OutSize[1], b3OutSize[2], b3OutSize[3]))
            print('maxPoolOutSize: %dx%dx%dx%d' % (maxPoolOutSize[0], maxPoolOutSize[1], maxPoolOutSize[2], maxPoolOutSize[3]))
            print('a4OutSize: %dx%dx%dx%d' % (a4OutSize[0], a4OutSize[1], a4OutSize[2], a4OutSize[3]))
            print('b4OutSize: %dx%dx%dx%d' % (b4OutSize[0], b4OutSize[1], b4OutSize[2], b4OutSize[3]))
            print('c4OutSize: %dx%dx%dx%d' % (c4OutSize[0], c4OutSize[1], c4OutSize[2], c4OutSize[3]))
            print('d4OutSize: %dx%dx%dx%d' % (d4OutSize[0], d4OutSize[1], d4OutSize[2], d4OutSize[3]))
            print('e4OutSize: %dx%dx%dx%d' % (e4OutSize[0], e4OutSize[1], e4OutSize[2], e4OutSize[3]))
            print('maxPoolOutSize: %dx%dx%dx%d' % (maxPool2OutSize[0], maxPool2OutSize[1], maxPool2OutSize[2], maxPool2OutSize[3]))
            print('a5OutSize: %dx%dx%dx%d' % (a5OutSize[0], a5OutSize[1], a5OutSize[2], a5OutSize[3]))
            print('b5OutSize: %dx%dx%dx%d' % (b5OutSize[0], b5OutSize[1], b5OutSize[2], b5OutSize[3]))
            print('avgPoolOutSize: %dx%dx%dx%d' % (avgPoolOutSize[0], avgPoolOutSize[1], avgPoolOutSize[2], avgPoolOutSize[3]))
            print('outputs: %dx%d' % (outputs.data.size()[0], outputs.data.size()[1]))
        '''
        #print('size of outputs: %dx%dx%dx%d' % (
        #    outputs.data.size()[0], outputs.data.size()[1], outputs.data.size()[2], outputs.data.size()[3]))
        # outputs contains 100 results, each result is a column vector [10x1]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'%(epoch+1, num_epochs, i+1,len(train_dataset)//batch_size, loss.data[0]))


# Change model to 'eval' mode (BN uses moving mean/var, stopping dropout).
cnn.eval()

correct = 0
total = 0
for images, labels in train_loader:
    images = to_var(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()
print 'Train Accuracy of the model on the %d train images: %d %%' % (train_dataset.train_labels.__len__(), 100 *correct / total)

correct = 0
total = 0
for images, labels in test_loader:
    images = to_var(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()
print 'Test Accuracy of the model on the %d test images: %d %%' % (test_dataset.test_labels.__len__(), 100 *correct / total)


# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn1.pkl')