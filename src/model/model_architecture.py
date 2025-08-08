import torch
import torchvision.models as models
from torch import nn
from torchsummary import summary
from torchvision.models import ResNet18_Weights


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, reduction)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

from torchvision.models.resnet import BasicBlock  # чтобы проверить тип

# Проход по всем слоям и замена BasicBlock на SEBasicBlock
def replace_basicblock_with_se(module, reduction=16):
    for i, (name, child) in enumerate(module.named_children()):
        if i <= 1:
            if isinstance(child, BasicBlock):
                # Заменяем только BasicBlock
                in_channels = child.conv1.in_channels
                out_channels = child.conv2.out_channels
                stride = child.conv2.stride[0] if hasattr(child.conv2, 'stride') else 1

                # Проверим, есть ли downsample
                downsample = child.downsample
                new_block = SEBasicBlock(in_channels, out_channels, stride=stride, reduction=reduction)
                new_block.conv1.weight.data = child.conv1.weight.data.clone()
                new_block.bn1.weight.data = child.bn1.weight.data.clone()
                new_block.bn1.bias.data = child.bn1.bias.data.clone()
                new_block.bn1.running_mean = child.bn1.running_mean.clone()  # ✅
                new_block.bn1.running_var = child.bn1.running_var.clone()  # ✅
                new_block.conv2.weight.data = child.conv2.weight.data.clone()
                new_block.bn2.weight.data = child.bn2.weight.data.clone()
                new_block.bn2.bias.data = child.bn2.bias.data.clone()
                new_block.bn2.running_mean = child.bn2.running_mean.clone()  # ✅
                new_block.bn2.running_var = child.bn2.running_var.clone()  # ✅

                if downsample is not None:
                    new_block.downsample[0].weight.data = downsample[0].weight.data.clone()
                    new_block.downsample[1].weight.data = downsample[1].weight.data.clone()
                    new_block.downsample[1].bias.data = downsample[1].bias.data.clone()
                    # ✅ Не забываем статистики BatchNorm в downsample
                    new_block.downsample[1].running_mean = child.downsample[1].running_mean.clone()
                    new_block.downsample[1].running_var = child.downsample[1].running_var.clone()

                setattr(module, name, new_block)

            else:
                # Рекурсивно заходим внутрь
                replace_basicblock_with_se(child, reduction)
        else:
            pass


def create_model(pretrained=True, freeze_backbone=True, reduction=8):
    """
    Создаёт и настраивает модель ResNet-18 с SE-блоками.

    Args:
        pretrained (bool): Загружать ли предобученные веса ImageNet.
        freeze_backbone (bool): Заморозить ли основную часть сети.
        reduction (int): Параметр сжатия в SE-слое.

    Returns:
        model (nn.Module): Готовая модель.
    """
    # Загружаем предобученную ResNet-18
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Заменяем BasicBlock на SEBasicBlock
    replace_basicblock_with_se(model, reduction=reduction)

    # Заменяем последний слой на 2 класса
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    if freeze_backbone:
        # Замораживаем все параметры
        for param in model.parameters():
            param.requires_grad = False

        # Размораживаем только layer4, SE-блоки и fc
        for param in model.layer4.parameters():
            param.requires_grad = True

        for name, module in model.named_modules():
            if isinstance(module, SELayer):
                for param in module.parameters():
                    param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    return model