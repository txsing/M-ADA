import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1= nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True)
        )
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True)
        )
        self.max_pool2 = nn.MaxPool2d(2, 2)

        self.fc1= nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(True)
        )
        self.fc2= nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True)
        )
        self.before_softmax = nn.Linear(1024, 10)

    def forward(self, x, return_feat=False):
        layers_output_dict = OrderedDict()

        x = self.conv1(x)
        layers_output_dict['conv1']=x
        
        x = self.max_pool1(x)
        layers_output_dict['max_pool1']=x
        
        x = self.conv2(x)
        layers_output_dict['conv2']=x
        
        x = self.max_pool2(x)
        layers_output_dict['max_pool2']=x
        
        x = self.fc1(x.view(x.size(0), -1))
        layers_output_dict['fc1']=x
        
        x = self.fc2(x)
        layers_output_dict['fc2']=x

        if return_feat:
            return x, self.before_softmax(x), layers_output_dict
        else:
            return self.before_softmax(x), layers_output_dict

class WAE(nn.Module):
    def __init__(self):
        super(WAE, self).__init__()

        self.fc1 = nn.Linear(3072, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 3072)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, 3072))
        return self.decode(z), z

class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=20):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z)