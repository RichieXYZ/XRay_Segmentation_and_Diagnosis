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


class UNetXRay(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        encoder_depth=5,
        decoder_channels=None,
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        assert encoder_depth == len(decoder_channels), \
            "encoder_depth must match length of decoder_channels"

        self.out_channels = out_channels
        self.depth = encoder_depth

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in decoder_channels[::-1]:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            decoder_channels[0],
            decoder_channels[0] * 2
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ch in decoder_channels:
            self.upconvs.append(
                nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
            )
            self.decoders.append(
                ConvBlock(ch * 2, ch)
            )

        # SR head
        self.final_conv = nn.Conv2d(
            decoder_channels[-1],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat([x, skips[i]], dim=1)
            x = self.decoders[i](x)

        # Binary Segmentation Head
        x = torch.sigmoid(self.final_conv(x))
        return x


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
    model = UNetXRay(
        in_channels=1,
        out_channels=1,
        encoder_depth=3,
        decoder_channels=[256, 128, 64],
    )
    print(count_params(model))
    print(model)

if __name__ == "__main__":
    main()