from .unet3d import UNet3D
from .dsa_unet3d import DSAUNet3D

__all__ = ["UNet3D", "DSAUNet3D", "build_model"]


def build_model(name, **kwargs):
    name = name.lower()
    if name in {"unet3d", "unet", "3d_unet"}:
        allowed = {"in_channels", "out_channels", "base_channels"}
        return UNet3D(**{k: v for k, v in kwargs.items() if k in allowed})
    if name in {
        "dsa_unet3d", "dsaunet3d", "dsa-unet3d", "dsa_unet", "dsa_unet3d_v2", "dsa_v2",
        "dsa_unet3d_gn", "dsa_gn", "dsa_unet3d_gn_no_attention", "dsa_gn_no_attention",
        "dsa_unet3d_hybrid", "dsa_hybrid"
    }:
        allowed = {
            "in_channels",
            "out_channels",
            "base_channels",
            "use_depthwise",
            "use_attention",
            "use_aspp",
            "norm_type",
            "residual_attention",
            "hybrid_depthwise",
        }
        if name in {"dsa_unet3d_v2", "dsa_v2"}:
            kwargs.setdefault("norm_type", "group")
            kwargs.setdefault("residual_attention", True)
        if name in {"dsa_unet3d_gn", "dsa_gn", "dsa_unet3d_gn_no_attention", "dsa_gn_no_attention"}:
            kwargs["norm_type"] = "group"
            kwargs["residual_attention"] = False
        if name in {"dsa_unet3d_gn_no_attention", "dsa_gn_no_attention"}:
            kwargs["use_attention"] = False
        if name in {"dsa_unet3d_hybrid", "dsa_hybrid"}:
            kwargs["hybrid_depthwise"] = True
        return DSAUNet3D(**{k: v for k, v in kwargs.items() if k in allowed})
    if name in {"swin_unetr", "swinunetr", "swin_unetr3d"}:
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError("SwinUNETR requires MONAI and einops.") from exc
        return SwinUNETR(
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1),
            feature_size=kwargs.get("swin_feature_size", 12),
            use_checkpoint=kwargs.get("swin_use_checkpoint", False),
            spatial_dims=3,
        )
    raise ValueError(f"Unknown model name: {name}")
