import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import math


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

class WAE_Cifar(nn.Module):
    def __init__(self, input_dims):
        super(WAE_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3= nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4= nn.Conv2d(32, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(18432, 1024) # 32 * 24 * 24
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 18432)
        self.bn8 = nn.BatchNorm1d(18432)
        
        self.conv5 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.bn10 = nn.BatchNorm2d(32)
        self.conv7= nn.ConvTranspose2d(32, 16, kernel_size=3)
        self.bn11 = nn.BatchNorm2d(16)
        self.conv8= nn.ConvTranspose2d(16, 3, kernel_size=3)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 18432) # 32 * 24 * 24
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        return x

    def decode(self, z):
        z = F.relu(self.bn7(self.fc3(z)))
        z = F.relu(self.bn8(self.fc4(z)))
        z = z.view(-1, 32, 24, 24)
        z = F.relu(self.bn9(self.conv5(z)))
        z = F.relu(self.bn10(self.conv6(z)))
        z = F.relu(self.bn11(self.conv7(z)))
        z = torch.sigmoid(self.conv8(z)) # why sigmoid?
        return z

    def forward(self, x):
        z = self.encode(x)
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

class WRN_BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WRN_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None
  
    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.is_in_equal_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.is_in_equal_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)

class WRN_NetworkBlock(nn.Module):
      """Layer container for blocks."""
    
      def __init__(self,
                   nb_layers,
                   in_planes,
                   out_planes,
                   block,
                   stride,
                   drop_rate=0.0):
        super(WRN_NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate)
    
      def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                      drop_rate):
        layers = []
        for i in range(nb_layers):
              layers.append(block(i == 0 and in_planes or out_planes, out_planes,i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    
      def forward(self, x):
        return self.layer(x)

class WRN_16_4(nn.Module):
    """WideResNet class."""

    def __init__(self):
        depth = 16
        num_classes = 10
        widen_factor = 4
        drop_rate = 0.0

        super(WRN_16_4, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = WRN_BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = WRN_NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                                   drop_rate)
        # 2nd block
        self.block2 = WRN_NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                                   drop_rate)
        # 3rd block
        self.block3 = WRN_NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                                   drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_feat=False):
        layers_output_dict = OrderedDict()

        out = self.conv1(x)
        layers_output_dict['layer0']=out
        out = self.block1(out)
        layers_output_dict['layer1']=out
        out = self.block2(out)
        layers_output_dict['layer2']=out
        out = self.block3(out)
        layers_output_dict['layer3']=out
        out = self.relu(self.bn1(out))
        layers_output_dict['layer4']=out
        out = F.avg_pool2d(out, 8)
        layers_output_dict['avg_pool']=out
        out = out.view(-1, self.n_channels)

        if return_feat:
            return x, self.fc(out), layers_output_dict
        else:
            return self.fc(out), layers_output_dict
