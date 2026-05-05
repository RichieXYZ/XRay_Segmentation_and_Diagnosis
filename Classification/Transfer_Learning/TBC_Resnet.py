import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# --- SE BLOCK ---
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# --- MAIN MODEL ---
class TBC_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv for grayscale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)

        # Use rest of backbone
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Attention
        self.se = SEBlock(512)

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.se(x)  # attention

        x = self.pool(x)
        x = self.classifier(x)

        return x  # logits

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
    model = TBC_ResNet18()
    print(count_params(model))
    print(model)

    model.eval()
    input = torch.rand(size=(1,1,256,256))
    out = model(input)
    print(out.shape)

if __name__ == "__main__":
    main()