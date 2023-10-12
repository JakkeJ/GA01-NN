import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling'
]


class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = "same"),
            nn.ReLU()
        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(1024, 604)
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        #nn.Flatten()
        x = self.temporal_pool(x)
        
        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    model = PHOSCnet()

    x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250).to('mps')

    y = model(x)

    #print(y['phos'].shape)
    #print(y['phoc'].shape)
