import torch
from modules.layers import *
import re
import torch.nn as nn
import copy
import types
__all__ = ['clrp_others', 'clrp_target', 'sglrp_others', 'sglrp_target', 'normalize']


# CLRP
def clrp_target(output, vis_class, **kwargs):
    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.zeros_like(output)
        mask.scatter_(1, pred, 1)
    elif vis_class == 'index':
        mask = torch.zeros_like(output)
        mask[:, kwargs['class_id']] = 1
    elif vis_class == 'target':
        mask = torch.zeros_like(output)
        mask.scatter_(1, kwargs['target'], 1)
    else:
        raise Exception('Invalid vis-class')

    return mask * output


def clrp_others(output, vis_class, **kwargs):
    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.ones_like(output)
        mask.scatter_(1, pred, 0)
        pred_out = output.gather(1, pred)
    elif vis_class == 'index':
        mask = torch.ones_like(output)
        mask[:, kwargs['class_id']] = 0
        pred_out = output[:, kwargs['class_id']:kwargs['class_id'] + 1]
    elif vis_class == 'target':
        mask = torch.ones_like(output)
        mask.scatter_(1, kwargs['target'], 0)
        pred_out = output.gather(1, kwargs['target'])
    else:
        raise Exception('Invalid vis-class')

    mask /= (output.shape[-1] - 1)

    return mask * pred_out


# SGLRP
def sglrp_target(output, vis_class, **kwargs):
    sm_pred = torch.softmax(output, dim=1)

    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.zeros_like(output)
        mask.scatter_(1, pred, 1)
    elif vis_class == 'index':
        mask = torch.zeros_like(output)
        mask[:, kwargs['class_id']] = 1
    elif vis_class == 'target':
        mask = torch.zeros_like(output)
        mask.scatter_(1, kwargs['target'], 1)
    else:
        raise Exception('Invalid vis-class')

    return mask * (sm_pred * (1 - sm_pred) + 1e-8)


def sglrp_others(output, vis_class, **kwargs):
    sm_pred = torch.softmax(output, dim=1)

    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.ones_like(output)
        mask.scatter_(1, pred, 0)
        sm_pred_out = sm_pred.gather(1, pred)
    elif vis_class == 'index':
        mask = torch.ones_like(output)
        mask[:, kwargs['class_id']] = 0
        sm_pred_out = sm_pred[:, kwargs['class_id']]
    elif vis_class == 'target':
        mask = torch.ones_like(output)
        mask.scatter_(1, kwargs['target'], 0)
        sm_pred_out = sm_pred.gather(1, kwargs['target'])
    else:
        raise Exception('Invalid vis-class')

    return mask * sm_pred_out * sm_pred


def normalize(tensor,
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

def _load_caffe_resnet50(model, checkpoint, make_bn_positive=False):
    # Patch the torchvision model to match the Caffe definition.
    model.conv1 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=True)
    model.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=0,
                                 ceil_mode=True)
    for i in range(2, 5):
        getattr(model, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(model, 'layer%d' % i)[0].conv2.stride = (1, 1)

    # Patch the checkpoint dict and load it.
    def rename(name):
        name = re.sub(r'bn(\d).(0|1).(.*)', r'bn\1.\3', name)
        name = re.sub(r'downsample.(\d).(0|1).(.*)', r'downsample.\1.\3', name)
        return name

    checkpoint = {rename(k): v for k, v in checkpoint.items()}

    # Convert from BGR to RGB.
    checkpoint['conv1.weight'] = checkpoint['conv1.weight'][:, [2, 1, 0], :, :]

    model.load_state_dict(checkpoint)

    # For EBP: the signs of the linear BN weights should be positive.
    # In practice there is only a tiny fraction of neg weights
    # and this does not seem to affect the results much.
    if make_bn_positive:
        conv = None
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                sign = module.weight.sign()
                module.weight.data *= sign
                module.running_mean.data *= sign
                conv.weight.data *= sign.view(-1, 1, 1, 1)
                if conv.bias is not None:
                    conv.bias.data *= sign
            conv = module

    _fix_caffe_maxpool(model)
    return model

def _caffe_resnet50_to_fc(model):
    # Shallow copy.
    model_ = copy.copy(model)

    # Patch the last layer: fc -> conv.
    out_ch, in_ch = model.fc.weight.shape
    conv = Conv2d(in_ch, out_ch, (1, 1))
    conv.weight.data.copy_(model.fc.weight.view(conv.weight.shape))
    conv.bias.data.copy_(model.fc.bias)
    model_.fc = conv

    # Patch average pooling.
    # model_.avgpool = nn.AvgPool2d((7, 7), stride=1, ceil_mode=True)

    def forward(self, x):
        # Same as original, but skip flatten layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    model_.forward = types.MethodType(forward, model_)
    return model_

def _fix_caffe_maxpool(model):
    for module in model.modules():
        if isinstance(module, torch.nn.MaxPool2d):
            module.ceil_mode = True