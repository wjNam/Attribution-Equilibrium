from typing import Any

import torch
import torch.nn as nn
from modules.layers import *
from modules.incept_utils import _log_api_usage_once
from modules._internally_replaced_utils import load_state_dict_from_url


__all__ = ["AlexNet", "alexnet"]


model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = Sequential(
            Dropout(p=dropout),
            Linear(256 * 6 * 6, 4096),
            ReLU(inplace=True),
            Dropout(p=dropout),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def hedge(self, R, flag=None, ratio=1):
        R = self.features.hedge(R, ratio)
        return R

def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
    return model