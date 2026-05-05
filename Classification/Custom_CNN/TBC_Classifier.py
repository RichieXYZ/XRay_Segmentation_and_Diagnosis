import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TBC_Classifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        encoder_depth=5,
        encoder_channels=None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256]

        assert encoder_depth == len(encoder_channels)

        self.depth = encoder_depth

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            encoder_channels[-1],
            encoder_channels[-1] * 2
        )

        bottleneck_channels = encoder_channels[-1] * 2

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # binary output (logit)
        )

    def forward(self, x):
        # Encoder
        for enc in self.encoders:
            x = enc(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        # Global Average Pooling
        x = self.global_pool(x)  # [B, C, 1, 1]
        # Classifier
        x = self.classifier(x)   # [B, 1]

        return x  # logits (NO sigmoid here)


class TBC_Classifier_v2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        encoder_depth=5,
        encoder_channels=None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256]

        assert encoder_depth == len(encoder_channels)

        self.depth = encoder_depth

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            encoder_channels[-1],
            encoder_channels[-1] * 2
        )

        bottleneck_channels = encoder_channels[-1] * 2

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # Encoder
        for enc in self.encoders:
            x = enc(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        # Global Average Pooling
        x = self.global_pool(x)  # [B, C, 1, 1]
        # Classifier
        x = self.classifier(x)   # [B, 1]

        return x  # logits (NO sigmoid here)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_params(model):
    import operator
    from functools import reduce
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def main():
    model = TBC_Classifier(in_channels=1,
        encoder_depth=3,
        encoder_channels=[256, 128, 64],
    )
    print(count_params(model))
    print(model)

    model.eval()
    input = torch.rand(size=(1,1,256,256))
    out = model(input)
    print(out.shape)

if __name__ == "__main__":
    main()