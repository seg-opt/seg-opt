from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.segmentation import SemanticSegmenter


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.convolutions(torch.cat((inputs, skip), dim=1))


class ResNet34UNet(SemanticSegmenter):
    """ResNet34 encoder with a four-stage skip-connected U-Net decoder."""

    def __init__(self, encoder: nn.Module, class_names: Sequence[str]) -> None:
        super().__init__(class_names)
        encoder.avgpool = nn.Identity()
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        self.classifier = nn.Conv2d(32, self.num_classes, kernel_size=1)
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        modules = (
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.classifier,
        )
        for parent in modules:
            for module in parent.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        stem = self.encoder.relu(
            self.encoder.bn1(self.encoder.conv1(pixel_values))
        )
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)

        decoded = self.decoder4(layer4, layer3)
        decoded = self.decoder3(decoded, layer2)
        decoded = self.decoder2(decoded, layer1)
        decoded = self.decoder1(decoded, stem)
        logits = self.classifier(decoded)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
