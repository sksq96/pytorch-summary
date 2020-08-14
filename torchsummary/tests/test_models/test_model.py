import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SingleInputNet(nn.Module):
    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNet(nn.Module):
    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    def __init__(self):
        super(MultipleInputNetDifferentDtypes, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.FloatTensor)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        # set x2 to FloatTensor
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class NestedNet(nn.Module):
    def __init__(self):
        super(NestedNet, self).__init__()
        self.conv_block1 = ConvBlock(1, 10, 5)
        self.conv_block2 = ConvBlock(10, 20, 5)
        self.conv_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv_block1(x))
        x = F.relu((self.conv_drop(self.conv_block2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.pool(x)
        return x


class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        weight_tensor = torch.rand(50, 50)
        self.W = Parameter(weight_tensor, requires_grad=True)

    def forward(self, x):
        return torch.einsum("bij,jk->bik", x, self.W)
