import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# Hyper Parameters
num_epochs = 50
batch_size = 100
learning_rate = 0.001

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
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.fc = nn.Linear(8*8*32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


cnn = CNN()

if torch.cuda.is_available():
    cnn = cnn.cuda()

# convert all the weights tensors to cuda()
# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
print 'number of parameters: ', sum(param.numel() for param in cnn.parameters())

# Train:
for epoch in range(num_epochs):
    # at each epoch we train the net on all 50000 images, i.e on all the train_dataset.
    # a single optimization step is performed per batchSize images that the net has seen.
    # therefor, the number of optimization steps performed in a single epoch is 50000/batchSize
    for i, (images, labels) in enumerate(train_loader):
        images = images.sum(dim=1)
        images = images[:, np.newaxis, :, :]
        # at images if shuffle is set to True the order of the images changes
        images = to_var(images)
        labels = to_var(labels)
        # images contains batchSize images, each image is a matrix [3x32x32]

        #imshow(torchvision.utils.make_grid(images.data / images.data.max()), "i: %d" %i).show()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
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
for images, labels in test_loader:
    images = images.sum(dim=1)
    images = images[:, np.newaxis, :, :]

    images = to_var(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()
print 'Test Accuracy of the model on the 10000 test images: %d %%' % (100 *correct / total)


# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')