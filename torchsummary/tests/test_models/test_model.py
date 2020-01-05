import torch
import torch.nn as nn
import torch.nn.functional as F

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
