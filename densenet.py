import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from cfg import *
from yolo_layer import YoloLayer
# you need to download the models to ~/.torch/models
# model_urls = {
#     'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
#     'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
#     'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
#     'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
# }

densenet121_model_name = 'densenet121-a639ec97.pth'
densenet169_model_name = 'densenet169-b2777c0a.pth'
densenet201_model_name = 'densenet201-c1103571.pth'
densenet161_model_name = 'densenet161-8d451a50.pth'
models_dir = os.path.expanduser(r'D:\liuyan2021\pth')


def densenet121(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet121_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if 'norm5' in key:
                continue
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


def densenet169(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet169_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet201_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet161_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate,
                                  kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

# from torchvision.models.resnet import
class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class AuxiliaryBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AuxiliaryBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=1, bias=False)

class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_output=75):

        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        for i in range(2, 5):
            self.add_module("smooth_layer%d"%(i), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        for i in range(2, 5):
            self.add_module("last_layer%d" % (i), nn.Conv2d(256, num_output, kernel_size=1, stride=1))

        self.auxiliary = nn.Sequential()
        self.lateral_block = nn.Sequential()
        num_features = num_init_features
        auxiliary_feature = num_init_features
        input_feature = num_init_features
        for i, num_layers in enumerate(block_config):
            auxiliary_feature = (auxiliary_feature + num_layers * growth_rate)//2
            self.auxiliary.add_module("auxiliaryblock%d" % (i + 1),
                                      AuxiliaryBlock(input_feature, auxiliary_feature, stride=2, downsample=nn.Sequential(conv3x3(input_feature, auxiliary_feature, 2), nn.BatchNorm2d(auxiliary_feature))))
            if i>=1:
                self.lateral_block.add_module("lateral_conv%d"% (i + 1), nn.Conv2d(auxiliary_feature, 256, kernel_size=1, stride=1))
            input_feature = auxiliary_feature
        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.trans = _Transition(num_input_features=num_features, num_output_features=num_features//2)
        self.features.add_module('transition%d' % (len(block_config) + 1), self.trans)
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        features = self.features[:3](x)
        trans1 = self.features.transition1(self.features.denseblock1(features))  # + self.auxiliary.auxiliaryblock1(features)
        trans2 = self.features.transition2(self.features.denseblock2(trans1))    # + self.auxiliary.auxiliaryblock2(trans1)
        trans3 = self.features.transition3(self.features.denseblock3(trans2))    # + self.auxiliary.auxiliaryblock3(trans2)
        trans4 = self.features.transition5(self.features.denseblock4(trans3))
        c4 = self.smooth_layer4(self.lateral_block.lateral_conv4(trans4))
        p4 = self.last_layer4(c4)
        c3 = self._upsample_add(c4, self.smooth_layer3(self.lateral_block.lateral_conv3(trans3)))
        p3 = self.last_layer3(c3)
        p2 = self.last_layer2(self._upsample_add(c3, self.smooth_layer2(self.lateral_block.lateral_conv2(trans2))))
        return p2, p3, p4

class DensnetYolo(nn.Module):
    def net_name(self):
        names_list = ('region', 'yololayer')
        name = names_list[0]
        for m in self.models:
            if isinstance(m, YoloLayer):
                name = names_list[1]
        return name

    def getLossLayers(self):
        loss_layers = []
        for m in self.models:
            if isinstance(m, YoloLayer):
                loss_layers.append(m)
        return loss_layers

    def __init__(self, cfgfile, use_cuda=True):
        super(DensnetYolo, self).__init__()
        self.use_cuda = use_cuda
        self.blocks = parse_cfg(cfgfile)
        self.seen = 0

        self.num_classes = int(self.blocks[1]['classes'])
        self.models = self.create_network(self.blocks)
        num_output = (5 + self.num_classes) * 3
        print("num_output = ", num_output)
        self.body = densenet121(pretrained=False, num_output=num_output)
        self.loss_layers = self.getLossLayers()

    def create_network(self, blocks):
        models = nn.ModuleList()
        for i, block in enumerate(blocks):
            if block['type'] == 'net':
                self.width = int(block['width'])
                self.height = int(block['height'])
                continue
            elif block['type'] == 'densenet':
                yololayer = YoloLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yololayer.anchor_mask = [int(i) for i in anchor_mask]
                yololayer.anchors = [float(i) for i in anchors]
                yololayer.num_classes = int(block['classes'])
                yololayer.num_anchors = int(block['num'])
                yololayer.anchor_step = len(yololayer.anchors)//yololayer.num_anchors
                try:
                    yololayer.rescore = int(block['rescore'])
                except:
                    pass
                yololayer.nth_layer = i
                yololayer.ignore_thresh = float(block['ignore_thresh'])
                yololayer.truth_thresh = float(block['truth_thresh'])
                yololayer.net_width = self.width
                yololayer.net_height = self.height
                models.append(yololayer)
        return models
    def forward(self, x):
        # x = self.model(x)
        # return x
        ind = -1
        outputs = dict()
        out_boxes = dict()
        outno = 0
        features = self.body(x)
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['densenet']:
                boxes = self.models[outno].get_mask_boxes(features[outno])
                out_boxes[outno] = boxes
                outno += 1
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x if outno == 0 else out_boxes
    def save_weight():
        pass
