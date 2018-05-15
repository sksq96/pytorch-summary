## Keras style `model.summary()` in PyTorch
[![PyPI version](https://badge.fury.io/py/torchsummary.svg)](https://badge.fury.io/py/torchsummary)

Keras has a neat API to view the visualization of the model which is very helpful while debugging your network. Here is a barebone code to try and mimic the same in PyTorch. The aim is to provide information complementary to, what is not provided by `print(your_model)` in PyTorch.

### Usage

- `git clone https://github.com/Bond-SYSU/pytorch-summary`

```python
from torchsummary import summary
summary(your_model, input_size=(channels, H, W))
```

- Note that the `input_size` is required to make a forward pass through the network.

### Examples

#### CNN for MNSIT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if torch.cuda.is_available():
    model.cuda()

summary(model, (1, 28, 28))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 24, 24]             260
            Conv2d-2             [-1, 20, 8, 8]            5020
         Dropout2d-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]           16050
            Linear-5                   [-1, 10]             510
================================================================
Total params: 21840
Trainable params: 21840
Non-trainable params: 0
----------------------------------------------------------------
```


#### VGG16


```python
import torch
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
if torch.cuda.is_available():
    vgg.cuda()

summary(vgg, (3, 224, 224))
```



```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]            1792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]           36928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]           73856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]          147584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]          295168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]          590080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]          590080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]         1180160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]         2359808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]         2359808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]         2359808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]         2359808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]         2359808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]       102764544
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]        16781312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]         4097000
================================================================
Total params: 138357544
Trainable params: 138357544
Non-trainable params: 0
----------------------------------------------------------------
```


### References

- The code is borrowed from [this PyTorch issue](https://github.com/pytorch/pytorch/issues/2001).
- Thanks to @ncullen93 and @HTLife. 

