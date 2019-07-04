## Keras style `model.summary()` in PyTorch
[![PyPI version](https://badge.fury.io/py/torchsummary.svg)](https://badge.fury.io/py/torchsummary)

Keras has a neat API to view the visualization of the model which is very helpful while debugging your network. Here is a barebone code to try and mimic the same in PyTorch. The aim is to provide information complementary to, what is not provided by `print(your_model)` in PyTorch. (**New functionality**) The main function `summary` (`from torchsummary import summary`) can also be used to infer the output shape of a pytorch model. Thus, it provides a way to build pytorch model that supports any input shape like in Keras (see an [example](#scalable) below).

### Usage

- `pip install torchsummary` or 
- `git clone https://github.com/sksq96/pytorch-summary`

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

summary(model, (1, 28, 28))
```


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 24, 24]             260
            Conv2d-2             [-1, 20, 8, 8]           5,020
         Dropout2d-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
```


#### VGG16


```python
import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

summary(vgg, (3, 224, 224))
```


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]     102,764,544
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.59
Params size (MB): 527.79
Estimated Total Size (MB): 746.96
----------------------------------------------------------------
```


#### Multiple Inputs


```python
import torch
import torch.nn as nn
from torchsummary import summary

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConv().to(device)

summary(model, [(1, 16, 16), (1, 28, 28)])
```


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 1, 16, 16]              10
              ReLU-2            [-1, 1, 16, 16]               0
            Conv2d-3            [-1, 1, 28, 28]              10
              ReLU-4            [-1, 1, 28, 28]               0
================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.77
Forward/backward pass size (MB): 0.02
Params size (MB): 0.00
Estimated Total Size (MB): 0.78
----------------------------------------------------------------
```

### Build pytorch model with scalable input shape (like Keras)<a name="scalable"></a>

```python

import torch
import torch.nn as nn
from torchsummary import summary

class AutoEncoder(nn.Module):
    """
    ResNet autoencoder network that support any input shape as model in Keras
    :param img_shape (tuple, channel last): support any image input shape 
    :param state_dim: (int) latent state dimension
    """

    def __init__(self, img_shape=(3, 224, 224), state_dim=3):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        ## Without torchsummary here, it's impossible to build model with scalable input shape as Keras.
        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.img_height * self.img_width * 64, state_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_height * self.img_width * 64)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2),
            nn.Tanh()
        )

    def encode(self, x):
        """
        Encode image to latent state
        """
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.encoder_fc(encoded)

    def decode(self, x):
        """
        Decode latent state to image
        """
        decoded = self.decoder_fc(x)
        decoded = decoded.view(x.size(0), 64, self.img_height, self.img_width)
        return self.decoder_conv(decoded)

    def forward(self, x):
        reconstruct = self.decode(self.encode(x))
        return reconstruct


img_shape = (3,128,128)
model = AutoEncoder(img_shape=img_shape, state_dim=100)
summary(model, img_shape)

```


### References

- The idea for this package sparked from [this PyTorch issue](https://github.com/pytorch/pytorch/issues/2001).
- Thanks to @ncullen93 and @HTLife. 
- For Model Size Estimation @jacobkimmel ([details here](https://github.com/sksq96/pytorch-summary/pull/21))
