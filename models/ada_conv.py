import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

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
    def __init__(self, input_dims):
        super(WAE, self).__init__()
        self.input_dims = input_dims
        # 3 * 222 * 222 = 147852
        # 3 * 32 * 32 = 3072
        self.fc1 = nn.Linear(self.input_dims, 400) 
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.input_dims)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dims))
        return self.decode(z), z

class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=20):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z)

class ResNet18(nn.Module):
    def __init__(self):
        block = BasicBlock
        layers = [2, 2, 2, 2]
        classes=7
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # stride 参数决定了 该 Residual Layer 的第一个 block 的 stride，进而决定了这个layer 会不会缩减图片尺寸
    # res18 里只有第一个 Layer 不需要缩减，其他 Layer 都需要减半尺寸
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # stride ！=1 意味着，第一个 block 需要缩减 size（第一个 block 的 X 和 Y 都需要进行 size 调整）， downsample 调整 X
        # block.expansion 是针对 Bottleneck Block，该类型 Block 会 4 倍地 expand input dimensions 
        # 所以其实 downsample 这里做了两件事 1. 调整 size  2. 调整 dimensions
        # 每一个 Layer 都有且只有一个 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 第一个 Block 用了 stride 参数，改变 size 且使用 downsample
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # 后面就默认为1，不改变 size

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, return_feat=False):
        layers_output_dict = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers_output_dict['layer0']=x
        x = self.maxpool(x)
        layers_output_dict['max_pool']=x
        x = self.layer1(x)
        layers_output_dict['layer1']=x
        x = self.layer2(x)
        layers_output_dict['layer2']=x
        x = self.layer3(x)
        layers_output_dict['layer3']=x
        x = self.layer4(x)
        layers_output_dict['layer4']=x

        x = self.avgpool(x)
        layers_output_dict['avg_pool']=x

        features = x
        x = x.view(x.size(0), -1)
        if return_feat:
            return features, self.class_classifier(x), layers_output_dict
        else:
            return self.class_classifier(x), layers_output_dict