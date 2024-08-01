import warnings
from collections import namedtuple
from typing import Callable, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules.layers import *
from modules.incept_utils import _log_api_usage_once
from modules._internally_replaced_utils import load_state_dict_from_url
# from .._internally_replaced_utils import load_state_dict_from_url
# from ..utils import _log_api_usage_once


__all__ = ["Inception3", "inception_v3", "InceptionOutputs", "_InceptionOutputs"]


model_urls = {
    # Inception v3 ported from TensorFlow
    "inception_v3_google": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
}

InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class Inception3(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(f"lenght of inception_blocks should be 7 instead of {len(inception_blocks)}")
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)
    def hedge(self, R, flag=None, ratio=1):
        # N x 2048 x 8 x 8
        R1 = self.Mixed_7c.hedge(R, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 2048 x 8 x 8
        R1 = self.Mixed_7b.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 1280 x 8 x 8
        R1 = self.Mixed_7a.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 768 x 17 x 17
        R1 = self.Mixed_6e.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 768 x 17 x 17
        R1 = self.Mixed_6d.hedge(R1, ratio)
        # print(R1.sum())
        # N x 768 x 17 x 17
        R1 = self.Mixed_6c.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 768 x 17 x 17
        R1 = self.Mixed_6b.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 768 x 17 x 17
        R1 = self.Mixed_6a.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 288 x 35 x 35
        R1 = self.Mixed_5d.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 288 x 35 x 35
        R1 = self.Mixed_5c.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 256 x 35 x 35
        R1 = self.Mixed_5b.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 192 x 35 x 35
        R1 = self.maxpool2.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 192 x 71 x 71
        R1 = self.Conv2d_4a_3x3.hedge(R1, ratio)
        # print(R1.sum(),R2.sum(),R3.sum(),R4.sum(),R1.max())
        # N x 80 x 73 x 73
        R1 = self.Conv2d_3b_1x1.hedge(R1, ratio)
        # print(R1.sum().sum(), R1.max().max().min())
        # N x 64 x 73 x 73
        R1 = self.maxpool1.hedge(R1, ratio)
        # print(R1.sum().sum(), R1.max().max().min())
        # N x 64 x 147 x 147
        R1 = self.Conv2d_2b_3x3.hedge(R1, ratio)
        # print(R1.sum(),R1.max())
        # N x 32 x 147 x 147
        R1 = self.Conv2d_2a_3x3.hedge(R1, ratio)
        # print(R1.sum().sum(), R1.max().max().min())
        # N x 32 x 149 x 149
        R1 = self.Conv2d_1a_3x3.hedge(R1, ratio)
        # print(R1.sum().sum(), R1.max().max().min())
        # N x 3 x 299 x 299

        return R1

class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

        self.avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.cat = Cat()
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat(outputs, 1)

    def hedge(self, R1, ratio):
        set1 = self.cat.hedge(R1, ratio)

        branch1x1_r1, branch5x5_r1, branch3x3dbl_r1, branch_pool_r1 = set1
        branch_pool_r1 = self.branch_pool.hedge(branch_pool_r1, ratio)
        x4_1 = self.avg_pool.hedge(branch_pool_r1, ratio)

        branch3x3dbl_r1 = self.branch3x3dbl_3.hedge(branch3x3dbl_r1, ratio)
        branch3x3dbl_r1 = self.branch3x3dbl_2.hedge(branch3x3dbl_r1, ratio)
        x3_1 = self.branch3x3dbl_1.hedge(branch3x3dbl_r1, ratio)

        branch5x5_r1 = self.branch5x5_2.hedge(branch5x5_r1, ratio)
        x2_1 = self.branch5x5_1.hedge(branch5x5_r1, ratio)

        x1_1 = self.branch1x1.hedge(branch1x1_r1, ratio)

        return x1_1 + x2_1 + x3_1 + x4_1
class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

        self.max_pool = MaxPool2d(kernel_size=3, stride=2)

        self.cat = Cat()
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.max_pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat(outputs, 1)

    def hedge(self, R1, ratio):
        set1 = self.cat.hedge(R1, ratio)
        branch3x3_r1, branch3x3dbl_r1, branch_pool_r1 = set1
        # x3 = self.max_pool.relprop(branch_pool, alpha)
        x3_1 = self.max_pool.hedge(branch_pool_r1, ratio)
        branch3x3dbl_r1 = self.branch3x3dbl_3.hedge(branch3x3dbl_r1, ratio)
        branch3x3dbl_r1 = self.branch3x3dbl_2.hedge(branch3x3dbl_r1, ratio)
        x2_1 = self.branch3x3dbl_1.hedge(branch3x3dbl_r1, ratio)

        x1_1 = self.branch3x3.hedge(branch3x3_r1, ratio)

        return x1_1 + x2_1 + x3_1
class InceptionC(nn.Module):
    def __init__(
        self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool_avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

        self.cat = Cat()
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat(outputs, 1)

    def hedge(self, R1, ratio):
        set1 = self.cat.hedge(R1, ratio)
        branch1x1_r1, branch7x7_r1, branch7x7dbl_r1, branch_pool_r1 = set1

        branch_pool_r1 = self.branch_pool.hedge(branch_pool_r1, ratio)
        x4_1 = self.branch_pool_avg_pool.hedge(branch_pool_r1, ratio)

        branch7x7dbl_r1 = self.branch7x7dbl_5.hedge(branch7x7dbl_r1, ratio)
        branch7x7dbl_r1 = self.branch7x7dbl_4.hedge(branch7x7dbl_r1, ratio)
        branch7x7dbl_r1 = self.branch7x7dbl_3.hedge(branch7x7dbl_r1, ratio)
        branch7x7dbl_r1 = self.branch7x7dbl_2.hedge(branch7x7dbl_r1, ratio)
        x3_1 = self.branch7x7dbl_1.hedge(branch7x7dbl_r1, ratio)

        branch7x7_r1 = self.branch7x7_3.hedge(branch7x7_r1, ratio)
        branch7x7_r1 = self.branch7x7_2.hedge(branch7x7_r1, ratio)
        x2_1 = self.branch7x7_1.hedge(branch7x7_r1, ratio)

        x1_1 = self.branch1x1.hedge(branch1x1_r1, ratio)

        return x1_1 + x2_1 + x3_1 + x4_1
class InceptionD(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
        self.max_pool = MaxPool2d(kernel_size=3, stride=2)
        self.cat = Cat()
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.max_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat(outputs, 1)
    def hedge(self, R1, ratio):
        set1 = self.cat.hedge(R1, ratio)
        branch3x3_r1, branch7x7x3_r1, branch_pool_r1 = set1
        x3_1 = self.max_pool.hedge(branch_pool_r1, ratio)

        branch7x7x3_r1 = self.branch7x7x3_4.hedge(branch7x7x3_r1, ratio)
        branch7x7x3_r1 = self.branch7x7x3_3.hedge(branch7x7x3_r1, ratio)
        branch7x7x3_r1 = self.branch7x7x3_2.hedge(branch7x7x3_r1, ratio)
        x2_1 = self.branch7x7x3_1.hedge(branch7x7x3_r1, ratio)
        branch3x3_r1 = self.branch3x3_2.hedge(branch3x3_r1, ratio)
        x1_1 = self.branch3x3_1.hedge(branch3x3_r1, ratio)

        return x1_1 + x2_1 + x3_1

class InceptionE(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.branch_pool_avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.cat1 = Cat()
        self.cat2 = Cat()
        self.cat3 = Cat()
    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = self.cat1(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = self.cat2(branch3x3dbl, 1)

        branch_pool = self.branch_pool_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.cat3(outputs, 1)
    def hedge(self, R1, ratio):
        branch1x1, branch3x3, branch3x3dbl, branch_pool = self.cat3.hedge(R1, ratio)

        branch_pool = self.branch_pool.hedge(branch_pool, ratio)
        x4 = self.branch_pool_avg_pool.hedge(branch_pool, ratio)

        branch3x3dbl_3a, branch3x3dbl_3b = self.cat2.hedge(branch3x3dbl, ratio)
        branch3x3dbl_3a = self.branch3x3dbl_3a.hedge(branch3x3dbl_3a, ratio)
        branch3x3dbl_3b = self.branch3x3dbl_3b.hedge(branch3x3dbl_3b, ratio)
        branch3x3dbl = branch3x3dbl_3a + branch3x3dbl_3b
        branch3x3dbl = self.branch3x3dbl_2.hedge(branch3x3dbl, ratio)
        x3 = self.branch3x3dbl_1.hedge(branch3x3dbl, ratio)

        branch3x3_2a, branch3x3_2b = self.cat1.hedge(branch3x3, ratio)
        branch3x3_2a = self.branch3x3_2a.hedge(branch3x3_2a, ratio)
        branch3x3_2b = self.branch3x3_2b.hedge(branch3x3_2b, ratio)
        branch3x3 = branch3x3_2a + branch3x3_2b
        x2 = self.branch3x3_1.hedge(branch3x3, ratio)

        x1 = self.branch1x1.hedge(branch1x1, ratio)

        return x1 + x2 + x3 + x4

class InceptionAux(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]
        self.avg_pool = AvgPool2d(kernel_size=5, stride=3)
        self.adaptive_avg_pool = AdaptiveAvgPool2d((1, 1))
    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = self.avg_pool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = self.adaptive_avg_pool(x)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x
    def hedge(self, R1, ratio):
        # N x 1000
        R1 = self.fc.hedge(R1, ratio)
        # N x 768
        R1 = R1.reshape_as(self.adaptive_avg_pool.Y)
        # N x 768 x 1 x 1
        R1 = self.adaptive_avg_pool.hedge(R1, ratio)
        # N x 768 x 1 x 1
        R1 = self.conv1.hedge(R1, ratio)
        # N x 128 x 5 x 5
        R1 = self.conv0.hedge(R1, ratio)
        # N x 768 x 5 x 5
        R1 = self.avg_pool.hedge(R1, ratio)
        # N x 768 x 17 x 17

        return R1

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)
        self.relu = ReLU()
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

    def hedge(self, R1, ratio):
        R1 = self.relu.hedge(R1, ratio)
        R1 = self.bn.hedge(R1, ratio)
        return self.conv.hedge(R1, ratio)
def inception_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Inception3:
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    The required minimum input size of the model is 75x75.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: True if ``pretrained=True``, else False.
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" in kwargs:
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
        else:
            original_aux_logits = True
        kwargs["init_weights"] = False  # we are loading weights from a pretrained model
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls["inception_v3_google"], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None
        return model

    return Inception3(**kwargs)