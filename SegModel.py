from typing import List

import torch
import torch.nn as nn

from base import SegmentationModel, SegmentationHead
from encoders import get_encoder
from decoders import UnetDecoder


class ProjectUNet(SegmentationModel):
    """ResNet-50 encoder + U-Net decoder with a projection head for contrast.

    ``forward(x)`` returns ``(logits, proj_feat)``; ``forward(x, out_features=True)``
    additionally returns an intermediate encoder feature map.
    """

    def __init__(
            self,
            encoder_name: str = 'resnet50',
            encoder_weight: object = 'imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs: object,
    ) -> object:
        super(SegmentationModel, self).__init__()

        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=encoder_weight, **kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )
        self.projector = nn.Sequential(
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(decoder_channels[1], 64, kernel_size=1)
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()


if __name__ == '__main__':
    x = torch.rand(2, 1, 224, 224).cuda()
    model = ProjectUNet('resnet50', None, classes=4, deep_stem=32).cuda()
    logits, proj = model(x)
    print('logits:', logits.shape, 'proj:', proj.shape)
    print('params:', sum(p.numel() for p in model.parameters()))
