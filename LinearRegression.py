import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Linear Regression:

# Hyper Params:
inputSize       = 1
outputSize      = 1
numEpochs       = 100000
learningRate    = 0.0001

# Toy Dataset
#xTrain = np.array([[3.3], [40.4], [50.5]] , dtype=np.float32)
xTrain = np.array(100*np.random.rand(50,1) , dtype=np.float32)
#xTrain = np.array([[3.3]] , dtype=np.float32)
yTrain = xTrain/5 + 30
yTrain = np.array(yTrain + 5*np.random.randn(yTrain.size,1), dtype=np.float32)

# Linear Regression Model:
class LinearRegression(nn.Module):
    def __init__(self,input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(inputSize,outputSize)

# Loss: mean square loss:
criterion = nn.MSELoss()

# Train the Model:
# convert numpy array to torch Variable
inputs  = Variable(torch.from_numpy(xTrain))
targets = Variable(torch.from_numpy(yTrain))

for epoch in range(numEpochs):

    # forward + backward + optimize:
    model.zero_grad()
    outputs = model(inputs)
    loss    = criterion(outputs,targets)
    # loss = (sum((outputs-targets)**2))/len(inputs)
    loss.backward()

    # update weights:
    paramNum = 0
    parametersValues = torch.Tensor([[0],[0]])
    parametersValuesGrads = torch.Tensor([[0], [0]])
    for p in model.parameters():
        #print('parameter no.%d value: %.4f' % (paramNum,p.data.numpy() ))
        parametersValues[paramNum] = p.data
        parametersValuesGrads[paramNum] = p.grad.data
        #print('indeed inputs*param - outputs = %.4f' % inputs*p.data - outputs)
        p.data.add_(-learningRate, p.grad.data)
        #p.data = p.data - learningRate*p.grad.data
        paramNum = paramNum + 1

    outputSanityCheck = ((inputs.data * parametersValues[0] + parametersValues[1]) - outputs.data).numpy()
    outputSanityCheck = np.abs(outputSanityCheck).max()
    #print('indeed max(abs((inputs*firstParam + secondParam))) = %.2f' % outputSanityCheck)

    if False:
        # lossSingleInput = (ax + b - target)^2
        # dlossSingleInput/da = 2*(ax + b - target)*x
        # dlossSingleInput/db = 2*(ax + b - target)
        gradWithRespectTo_a = 2*(inputs.data * parametersValues[0] + parametersValues[1] - targets.data)*inputs.data
        gradWithRespectTo_b = 2 * (inputs.data * parametersValues[0] + parametersValues[1] - targets.data)
        print('autograd for a: %.2f' % parametersValuesGrads[0].numpy())
        print('calculated grad for a: %.2f' % gradWithRespectTo_a.numpy())
        print('autograd for b: %.2f' % parametersValuesGrads[1].numpy())
        print('calculated grad for b: %.2f' % gradWithRespectTo_b.numpy())


    if (epoch+1) % 5 == 0:
        #print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, numEpochs, loss.data[0]))
        print('model: out = %.2f x input + %.2f' % (parametersValues[0].numpy(), parametersValues[1].numpy()))


torch.save(model.state_dict(), 'model.pkl')
