"""
member2_siamese/siamese_net.py
────────────────────────────────────────────────────────────────────────────
Member 2 – Siamese Change Detection Network

Architecture options:
  V1 (SiameseUNet)   – ResNet-50 encoder, simple concatenation fusion
  V2 (SiameseUNetV2) – ResNet-50 encoder, CBAM + diff fusion, optional deep supervision
  V3 (SiameseUNetV3) – Swin-T encoder, CBAM + diff fusion, optional deep supervision

Stage guidance:
  Stage 1 → Use SiameseUNet (V1)  — no CBAM, no deep supervision
  Stage 2 → Use SiameseUNetV2    — CBAM + diff fusion, deep_supervision=False
  Stage 3 → Use SiameseUNetV2    — same as Stage 2 (Lovász only changes the loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

# ── Helper blocks ─────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """3×3 Conv → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Bilinear upsample-to-skip-shape → concat skip → 2× ConvBnRelu."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── Shared Encoder ────────────────────────────────────────────────────────────

class ResNet50Encoder(nn.Module):
    """
    ResNet-50 backbone returning multi-scale features at 1/4, 1/8, 1/16, 1/32.
    The stem conv is patched to accept in_channels != 3.
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        if in_channels != 3:
            old = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight
                mean_w = old.weight.mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_conv.weight[:, i:i+1] = mean_w
            backbone.conv1 = new_conv

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool   = backbone.maxpool
        self.layer1 = backbone.layer1   # → (B, 256,  H/4,  W/4)
        self.layer2 = backbone.layer2   # → (B, 512,  H/8,  W/8)
        self.layer3 = backbone.layer3   # → (B, 1024, H/16, W/16)
        self.layer4 = backbone.layer4   # → (B, 2048, H/32, W/32)

    def forward(self, x: torch.Tensor):
        x  = self.pool(self.layer0(x))
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e1, e2, e3, e4    # (256, 512, 1024, 2048)


# ── Models ────────────────────────────────────────────────────────────────────

class SiameseUNet(nn.Module):
    """
    V1: Siamese Change Detection Network (ResNet-50, simple concatenation fusion).

    This is the STABLE BASELINE for Stage 1.
    No CBAM, no deep supervision — straightforward Dice+BCE training.
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained)

        self.dec4 = DecoderBlock(in_ch=4096, skip_ch=2048, out_ch=256)
        self.dec3 = DecoderBlock(in_ch=256,  skip_ch=1024, out_ch=128)
        self.dec2 = DecoderBlock(in_ch=128,  skip_ch=512,  out_ch=64)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _fuse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)

        f1 = self._fuse(p1, q1)   # (B, 512,  H/4)
        f2 = self._fuse(p2, q2)   # (B, 1024, H/8)
        f3 = self._fuse(p3, q3)   # (B, 2048, H/16)
        f4 = self._fuse(p4, q4)   # (B, 4096, H/32)

        x = self.dec4(f4, f3)    # (B, 256, H/16)
        x = self.dec3(x,  f2)    # (B, 128, H/8)
        x = self.dec2(x,  f1)    # (B, 64,  H/4)
        return self.head(x)       # (B, 1,   H)


# ── CBAM Attention Modules (used in V2 and V3) ───────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = x.mean(dim=[2, 3])                           # (B, C)
        mx  = x.amax(dim=[2, 3])                           # (B, C)
        attn = torch.sigmoid(self.fc(avg) + self.fc(mx))   # (B, C)
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        mx   = x.amax(dim=1, keepdim=True)                 # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class FusionHead(nn.Module):
    """
    Takes cat(a, b, |a-b|, a*b) = 4C channels.
    Reduces to 2C then applies CBAM attention.
    """
    def __init__(self, C: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(4 * C, 2 * C, 1, bias=False),
            nn.BatchNorm2d(2 * C),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(2 * C)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([a, b, (a - b).abs(), a * b], dim=1)  # (B, 4C, H, W)
        return self.cbam(self.reduce(fused))                      # (B, 2C, H, W)


# ── SiameseUNetV2 ─────────────────────────────────────────────────────────────

class SiameseUNetV2(nn.Module):
    """
    Siamese Change Detection V2.
    Improvements over V1:
    - Difference fusion: cat(a, b, |a-b|, a*b) instead of just cat(a, b)
    - CBAM attention at each fusion scale
    - Optional deep supervision auxiliary heads

    Parameters
    ----------
    in_channels      : channels per image (3=RGB, 5=RGB+NDVI+NDWI)
    pretrained       : use ImageNet weights for encoder
    deep_supervision : if True, forward() returns [main, aux4, aux3] during training
                       if False, returns just main logits tensor
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained)

        # FusionHead at each scale: input is 4×encoder_channels, output is 2×encoder_channels
        # encoder channels: e1=256, e2=512, e3=1024, e4=2048
        self.fuse1 = FusionHead(256)    # output: 512
        self.fuse2 = FusionHead(512)    # output: 1024
        self.fuse3 = FusionHead(1024)   # output: 2048
        self.fuse4 = FusionHead(2048)   # output: 4096

        # Decoder — same structure as V1 because fusion output channels match V1
        self.dec4 = DecoderBlock(in_ch=4096, skip_ch=2048, out_ch=256)
        self.dec3 = DecoderBlock(in_ch=256,  skip_ch=1024, out_ch=128)
        self.dec2 = DecoderBlock(in_ch=128,  skip_ch=512,  out_ch=64)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Auxiliary heads for deep supervision (only used during training if enabled)
        if deep_supervision:
            self.aux4 = nn.Conv2d(256, 1, kernel_size=1)   # after dec4
            self.aux3 = nn.Conv2d(128, 1, kernel_size=1)   # after dec3

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)

        # Attention-gated difference fusion at each scale
        f1 = self.fuse1(p1, q1)   # (B, 512,  H/4)
        f2 = self.fuse2(p2, q2)   # (B, 1024, H/8)
        f3 = self.fuse3(p3, q3)   # (B, 2048, H/16)
        f4 = self.fuse4(p4, q4)   # (B, 4096, H/32)

        x4 = self.dec4(f4, f3)    # (B, 256, H/16)
        x3 = self.dec3(x4, f2)    # (B, 128, H/8)
        x2 = self.dec2(x3, f1)    # (B, 64,  H/4)

        main = self.head(x2)       # (B, 1, H, W)

        if self.deep_supervision and self.training:
            # Upsample aux outputs to full resolution for loss computation
            aux4 = F.interpolate(self.aux4(x4), size=main.shape[-2:],
                                 mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.aux3(x3), size=main.shape[-2:],
                                 mode="bilinear", align_corners=False)
            return [main, aux4, aux3]

        return main


# ── SiameseUNetV3 ─────────────────────────────────────────────────────────────

class SiameseUNetV3(nn.Module):
    """
    V3: Swin-T encoder with advanced diff fusion, CBAM attention, and Deep Supervision.
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Swin-T features: [96, 192, 384, 768] at strides 4, 8, 16, 32
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )

        self.fuse1 = FusionHead(96)
        self.fuse2 = FusionHead(192)
        self.fuse3 = FusionHead(384)
        self.fuse4 = FusionHead(768)

        # Decoder dimensions correspond to 2*C for each scale
        self.dec4 = DecoderBlock(in_ch=1536, skip_ch=768, out_ch=384)
        self.dec3 = DecoderBlock(in_ch=384,  skip_ch=384, out_ch=192)
        self.dec2 = DecoderBlock(in_ch=192,  skip_ch=192, out_ch=96)

        if self.deep_supervision:
            self.aux4 = nn.Conv2d(384, 1, kernel_size=1)
            self.aux3 = nn.Conv2d(192, 1, kernel_size=1)

        # Swin-T begins outputting at stride=4, so map back to original space
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBnRelu(96, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)

        f1 = self.fuse1(p1, q1)
        f2 = self.fuse2(p2, q2)
        f3 = self.fuse3(p3, q3)
        f4 = self.fuse4(p4, q4)

        d4 = self.dec4(f4, f3)
        d3 = self.dec3(d4,  f2)
        d2 = self.dec2(d3,  f1)
        main_logits = self.head(d2)

        if self.deep_supervision:
            a4 = self.aux4(d4)
            a3 = self.aux3(d3)
            return [main_logits, a3, a4]
        return main_logits
