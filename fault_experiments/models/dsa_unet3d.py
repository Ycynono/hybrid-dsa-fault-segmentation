import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization3d(channels, norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    if norm_type == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization type: {norm_type}")


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def conv3d(in_channels, out_channels, use_depthwise=True, kernel_size=3, padding=1, dilation=1):
    if use_depthwise:
        return DepthwiseSeparableConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool3d(x, 1)
        maxv = F.adaptive_max_pool3d(x, 1)
        weight = torch.sigmoid(self.mlp(avg) + self.mlp(maxv))
        return x * weight


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.amax(x, dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * weight


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=8, residual=False):
        super().__init__()
        self.channel = ChannelAttention3D(channels, reduction=reduction)
        self.spatial = SpatialAttention3D()
        self.residual = residual

    def forward(self, x):
        attended = self.spatial(self.channel(x))
        return x + attended if self.residual else attended


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=True,
        use_attention=True,
        norm_type="batch",
        residual_attention=False,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3d(in_channels, out_channels, use_depthwise=use_depthwise),
            normalization3d(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv3d(out_channels, out_channels, use_depthwise=use_depthwise),
            normalization3d(out_channels, norm_type),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                normalization3d(out_channels, norm_type),
            )
        self.attention = (
            CBAM3D(out_channels, residual=residual_attention) if use_attention else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out = self.relu(out + residual)
        return self.attention(out)


class ASPP3D(nn.Module):
    def __init__(
        self, in_channels, out_channels, use_depthwise=True, rates=(1, 2, 4, 6), norm_type="batch"
    ):
        super().__init__()
        branch_channels = max(out_channels // (len(rates) + 1), 1)
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    conv3d(
                        in_channels,
                        branch_channels,
                        use_depthwise=use_depthwise,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                    ),
                    normalization3d(branch_channels, norm_type),
                    nn.ReLU(inplace=True),
                )
            )

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        concat_channels = branch_channels * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=1, bias=False),
            normalization3d(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        features = [branch(x) for branch in self.branches]
        global_feature = self.global_branch(x)
        global_feature = F.interpolate(global_feature, size=size, mode="trilinear", align_corners=False)
        features.append(global_feature)
        return self.project(torch.cat(features, dim=1))


class DSAUNet3D(nn.Module):
    """Depthwise-separable attention U-Net for 3D seismic fault segmentation.

    Module switches are intentionally exposed for ablation studies.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=8,
        use_depthwise=True,
        use_attention=True,
        use_aspp=True,
        norm_type="batch",
        residual_attention=False,
        hybrid_depthwise=False,
    ):
        super().__init__()
        c = base_channels
        deep_block_args = {
            "use_depthwise": use_depthwise,
            "use_attention": use_attention,
            "norm_type": norm_type,
            "residual_attention": residual_attention,
        }
        shallow_block_args = dict(deep_block_args)
        if hybrid_depthwise and use_depthwise:
            shallow_block_args["use_depthwise"] = False
        self.enc1 = ResidualBlock3D(in_channels, c, **shallow_block_args)
        self.enc2 = ResidualBlock3D(c, c * 2, **shallow_block_args)
        self.enc3 = ResidualBlock3D(c * 2, c * 4, **deep_block_args)
        self.enc4 = ResidualBlock3D(c * 4, c * 8, **deep_block_args)

        if use_aspp:
            self.bottleneck = ASPP3D(
                c * 8, c * 8, use_depthwise=use_depthwise, norm_type=norm_type
            )
        else:
            self.bottleneck = ResidualBlock3D(c * 8, c * 8, **deep_block_args)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(c * 8, c * 4, **deep_block_args)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(c * 4, c * 2, **shallow_block_args)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(c * 2, c, **shallow_block_args)
        self.out = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
