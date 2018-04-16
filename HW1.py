import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


# Problem params:
inputSize       = 784
numClasses      = 10

# Hyper params:
numEpochs       = 40
batchSize       = 100
learningRate    = 0.01
nNeuronsInFirstLayer = 72

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

'''
We would like the first layer to give correlations to geometric shapes.
Every neuron gives the correlation for a specific shape in a specific rotation at a specific region of the image.
Assume we have 6 different rotation options and 6 different regions. Therefore we have 36 neurons per shape.
What shapes do we need?
- Quarter of a circle with small radius 
- Quarter of a circle with big radius
- small line
Therefore 3 shapes. 
How many neurons at first layer? 3*36=108.
Input size is 28*28=784.
So 784+1 params per neuron. 
Total of 785*108~78500 parameters. 
We have a limitation of 65000 parameters. 
7850 parameters are for the second layer. 
So about 57000 for the first layer - > 72 neurons - > 24 per shape instead of desirable 36. Thats ok. Let's roll! 
'''

class LogisticRegression(nn.Module):
    def __init__(self,input_size, num_classes, nNeuronsInFirstLayer):
        super(LogisticRegression, self).__init__()
        self.firstLayer = nn.Linear(input_size, nNeuronsInFirstLayer)
        self.outLayer = nn.Linear(nNeuronsInFirstLayer,num_classes)

    def forward(self, x):
        firstLayerOut = F.relu(self.firstLayer(x))
        return self.outLayer(firstLayerOut)


# at http://yann.lecun.com/exdb/mnist/ you can read that:
# "The MNIST database of handwritten digits, available from this page,
#   has a training set of 60,000 examples, and a test set of 10,000 examples.
#   It is a subset of a larger set available from NIST.
#   The digits have been size-normalized and centered in a fixed-size image."

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

print('in train_dataset there are %d images' % train_dataset.train_labels.size())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batchSize,
                                           shuffle=True)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

print('in test_dataset there are %d images' % test_dataset.test_labels.size())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batchSize,
                                           shuffle=False)

model = LogisticRegression(inputSize, numClasses, nNeuronsInFirstLayer)
if torch.cuda.is_available():
    model = model.cuda()

# Loss and Optimizer
# Softmax is internally computed
# a good article at: http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

###     Training     ###

# to learn about enumerate see: https://docs.python.org/2.3/whatsnew/section-enumerate.html
for epoch in range(numEpochs):
    # at each epoch we train the net on all 60000 images, i.e on all the train_dataset.
    # a single optimization step is performed per batchSize images that the net has seen.
    # therefor, the number of optimization steps performed in a single epoch is 60000/batchSize
    for i, (images,labels) in enumerate(train_loader):
        #imshow(images[0], labels[0]).show()

        # at images if shuffle is set to True the order of the images changes
        images = to_var(images.view(-1,28*28))
        labels = to_var(labels)
        # images contains batchSize images, each image is a column vector [28*28,1]

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(images)
        # outputs contains 100 results, each result is a column vector [10x1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch+1, numEpochs, i+1, len(train_dataset)//batchSize, loss.data[0]))

###     Testing     ###

correct = 0
total   = 0

for images, labels in test_loader:
    images = to_var(images.view(-1, 28*28))
    # images has batchSize images
    outputs = model(images)
    # outputs has 10 values for each image
    _, predicted = torch.max(outputs.data, 1)
    # for each one of the batchSize images, predicted has the index of the max value from outputs
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()
    #break
print ('Accuracy of the model on the %d test images: %d %%' % (test_dataset.test_labels.size()[0],
                                                                   100 * correct / total))
print ('#Correct is: %d; #total is: %d' % (correct,total))

print('no. of parameters in the model is: %d' % sum(param.numel() for param in model.parameters()))

# save the model
torch.save(model.state_dict(), 'model.plk')

