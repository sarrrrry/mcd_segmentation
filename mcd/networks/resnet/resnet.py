#####################
import math
import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


class Conv3x3(nn.Conv2d):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__(in_planes, out_planes,
                         kernel_size=3, stride=stride,
                         padding=1, bias=False)


class BaseResBlock(nn.Module):
    pass


class BasicBlock(BaseResBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = SyncBatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = SyncBatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x.copy_()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


from torch.nn import BatchNorm2d


class Bottleneck(BaseResBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = SyncBatchNorm(planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = SyncBatchNorm(planes)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = SyncBatchNorm(planes * 4)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x.copy_()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


from typing import Tuple
from dataclasses import dataclass


@dataclass
class ResLayer:
    block: BaseResBlock
    layer: Tuple[int, int, int, int]


class ResGenerator(nn.Module):
    def __init__(self, reslayer: ResLayer):
        super().__init__()
        self.inplanes = 128
        block = reslayer.block
        layers = reslayer.layer

        self.conv1 = Conv3x3(3, 64, stride=2)
        # self.bn1 = SyncBatchNorm(64)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(64, 64)
        # self.bn2 = SyncBatchNorm(64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = Conv3x3(64, 128)
        # self.bn3 = SyncBatchNorm(128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # SyncBatchNorm(planes * block.expansion),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x


class ResClassifier(nn.Module):
    in_size = 1024

    def __init__(self, out_):
        super(ResClassifier, self).__init__()
        self.fc = nn.Linear(self.in_size, out_)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


from mcd.networks.base_net_mcd import BaseNetMCD


class ResNet(BaseNetMCD):
    NET18 = "net18"
    NET50 = "net50"
    NET101 = "net101"

    LAYERS = {
        NET18: ResLayer(BasicBlock, (2, 2, 2, 2)),
        NET50: ResLayer(Bottleneck, (3, 4, 6, 3)),
        NET101: ResLayer(Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(self, layer_name: str, num_classes: int = 1000):
        """
        :type layer_name: str
        :type num_classes: int
        """
        super(ResNet, self).__init__()

        if not layer_name in self.LAYERS.keys():
            raise ValueError

        layer = self.LAYERS[layer_name]
        self.generator = ResGenerator(layer)
        self.classifier_f1 = ResClassifier(512 * layer.block.expansion, num_classes)
        self.classifier_f2 = ResClassifier(512 * layer.block.expansion, num_classes)
        self.init_weight()

    def forward(self, x):
        x = self.generator(x)
        x_f1 = self.classifier_f1(x)
        x_f2 = self.classifier_f2(x)
        return x_f1, x_f2

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, SyncBatchNorm):
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                m.bias.data.fill_(0.01)


if __name__ == '__main__':
    device = torch.device("cuda")
    resnet = ResNet(ResNet.NET50).to(device)
    resnet = nn.DataParallel(resnet)
    img = torch.ones((2, 3, 256, 256)).to(device)
    ret = resnet(img)
    print(ret)
