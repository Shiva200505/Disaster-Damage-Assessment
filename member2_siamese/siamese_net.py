"""
member2_siamese/siamese_net.py
────────────────────────────────────────────────────────────────────────────
Member 2 – Siamese Change Detection Network

Architecture:
  • Shared ResNet-50 encoder (ImageNet pretrained) processes pre and post
    disaster images in parallel.
  • At each encoder scale, the two feature maps are concatenated
    (cat fusion: 2×C channels) to highlight cross-image differences.
  • A U-Net style decoder with skip connections upsamples back to the
    original spatial resolution and outputs a binary change mask.

Channel flow (H = input height, input channels = 3 by default):
  Encoder:  e1(256,H/4)  e2(512,H/8)  e3(1024,H/16)  e4(2048,H/32)
  Fusion:   f1(512,H/4)  f2(1024,H/8) f3(2048,H/16)  f4(4096,H/32)
  Decoder:
    dec4(f4, skip=f3): (4096+2048)→256  @H/16
    dec3(x,  skip=f2): ( 256+1024)→128  @H/8
    dec2(x,  skip=f1): ( 128+ 512)→ 64  @H/4
    head: upsample ×4 → 1ch mask        @H

Input  : pre_img  (B, C, H, W)
         post_img (B, C, H, W)
Output : change_logits (B, 1, H, W) — apply sigmoid for mask probability

Reference:
  FC-EF / Siamese-Diff (Daudt et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


# ── Siamese U-Net ─────────────────────────────────────────────────────────────

class SiameseUNet(nn.Module):
    """
    Siamese Change Detection Network.

    Both images share a single ResNet-50 encoder. At each scale their
    feature maps are concatenated (cat-fusion) producing 2×C channels.
    The decoder reconstructs a pixel-wise binary change mask.

    Decoder channel math:
        dec4 conv input: fused_e4(4096) + skip_fused_e3(2048) = 6144 → 256
        dec3 conv input: 256 + fused_e2(1024)                = 1280 → 128
        dec2 conv input: 128 + fused_e1( 512)                =  640 →  64
        head: ×4 upsample → 1-channel output

    Parameters
    ----------
    in_channels : channels per image (3=RGB, 5=RGB+NDVI+NDWI)
    pretrained  : use ImageNet weights for the ResNet-50 encoder
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained)

        # Fused channel sizes: 2 × encoder channel count
        # f4=4096, f3=2048, f2=1024, f1=512
        self.dec4 = DecoderBlock(in_ch=4096, skip_ch=2048, out_ch=256)
        self.dec3 = DecoderBlock(in_ch=256,  skip_ch=1024, out_ch=128)
        self.dec2 = DecoderBlock(in_ch=128,  skip_ch=512,  out_ch=64)

        # Head: ×4 upsample from H/4 → H (from pool stride-2 + layer0 stride-2)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _fuse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate two same-shape feature tensors → 2×C channels."""
        return torch.cat([a, b], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pre, post : (B, C, H, W) float tensors

        Returns
        -------
        logits : (B, 1, H, W) raw logits; apply sigmoid for [0,1] change probability
        """
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)

        # Cat-fuse at each scale: 2×encoder_channels
        f1 = self._fuse(p1, q1)   # (B, 512,  H/4)
        f2 = self._fuse(p2, q2)   # (B, 1024, H/8)
        f3 = self._fuse(p3, q3)   # (B, 2048, H/16)
        f4 = self._fuse(p4, q4)   # (B, 4096, H/32)

        # Decode coarse → fine using earlier fused features as skips
        x = self.dec4(f4, f3)    # (B, 256, H/16)
        x = self.dec3(x,  f2)    # (B, 128, H/8)
        x = self.dec2(x,  f1)    # (B, 64,  H/4)
        return self.head(x)       # (B, 1,   H)
