import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class FedResBlock(nn.Module):
    """
    Federe öğrenme için özelleştirilmiş residual blok.
    BatchNorm yerine GroupNorm kullanır (küçük/değişken batch'lerde daha kararlı).
    """
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedCIFAR10CNN(nn.Module):
    """
    Daha yüksek doğruluk için derinleştirilmiş ve FL'e uyumlu CIFAR-10 CNN.
    Residual bloklar + GroupNorm ile daha kararlı eğitim.
    """
    def __init__(self):
        super().__init__()
        self.in_planes = 32

        # Giriş katmanı
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=32)

        # Residual katmanlar
        self.layer1 = self._make_layer(32, num_blocks=2, stride=1)  # 32x32
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2) # 8x8
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2) # 4x4

        # Sınıflandırıcı
        self.linear = nn.Linear(256, 10)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(FedResBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 4x4 -> 1x1
        out = out.view(out.size(0), -1)  # [batch, 256]
        out = self.linear(out)  # [batch, 10]
        return out

def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """Recursively replace all BatchNorm2d/1d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(min(num_groups, num_channels), num_channels))
        else:
            replace_bn_with_gn(child, num_groups)
    return module


class ResNetCIFAR10(nn.Module):
    """
    CIFAR-10 için ResNet-18: ilk konvolüsyon 3x3/stride=1 ve maxpool kaldırıldı.
    """
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None, num_classes=10)
        # Adapt to CIFAR-10 32x32 inputs
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DPResNetCIFAR10(nn.Module):
    """
    DP-friendly ResNet-18 for CIFAR-10.
    BatchNorm → GroupNorm for stable DP training with small/varying batches.
    """
    def __init__(self, num_groups: int = 8):
        super().__init__()
        m = models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        replace_bn_with_gn(m, num_groups)
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet8CIFAR10(nn.Module):
    """
    DP-friendly ResNet-8 for CIFAR-10.

    He et al. (2016) 6n+2 CIFAR ResNet family with n=1 → 8 trainable layers
    (1 stem conv + 3 stages × 1 BasicBlock × 2 convs + final FC).
    Channels [16, 32, 64], strides [1, 2, 2] for spatial downsampling.

    Uses GroupNorm and non-inplace ReLU for compatibility with Opacus
    per-sample gradient hooks. ~78K parameters → ≈12x less DP noise norm
    than DPResNetCIFAR10 (~11.2M params) at the same sigma.
    """

    def __init__(self, num_groups: int = 8):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(num_groups, 16), num_channels=16)

        self.layer1 = self._make_layer(16, stride=1, num_groups=num_groups)
        self.layer2 = self._make_layer(32, stride=2, num_groups=num_groups)
        self.layer3 = self._make_layer(64, stride=2, num_groups=num_groups)

        self.linear = nn.Linear(64, 10)

    def _make_layer(self, planes: int, stride: int, num_groups: int) -> nn.Module:
        block = _CIFARResBlock(self.in_planes, planes, stride=stride, num_groups=num_groups)
        self.in_planes = planes
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class _CIFARResBlock(nn.Module):
    """Basic residual block for CIFAR ResNet family. GroupNorm + non-inplace ReLU."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, num_groups: int = 8):
        super().__init__()
        g_in = min(num_groups, planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=g_in, num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=g_in, num_channels=planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=g_in, num_channels=planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class EMAModel:
    """Exponential Moving Average of model parameters for smoother convergence under DP noise."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return dict(self.shadow)


if __name__ == "__main__":
    for cls in [ResNetCIFAR10, DPResNetCIFAR10, ImprovedCIFAR10CNN, ResNet8CIFAR10]:
        model = cls()
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{cls.__name__} -> output shape: {y.shape}, params: {n_params:,}")
