import torch
import torch.nn as nn

# Pytorch Models
class VGG(nn.Module):
    def __init__(self, input_channels=1, n_emotions=7, filters_levels=(64, 128, 256, 512), dense_units=4096, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, filters_levels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_levels[0], filters_levels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filters_levels[0], filters_levels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_levels[1], filters_levels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filters_levels[1], filters_levels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_levels[2], filters_levels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filters_levels[2], filters_levels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_levels[3], filters_levels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_levels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters_levels[3] * 3 * 3, dense_units),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, n_emotions),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=1, n_emotions=7):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_emotions)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18_pytorch(input_channels=1, n_emotions=7):
    return ResNet(ResidualBlock, [2, 2, 2, 2], input_channels=input_channels, n_emotions=n_emotions)
