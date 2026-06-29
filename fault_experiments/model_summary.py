import argparse
import json

import torch

from fault_experiments.models import build_model


def summarize_model(model, input_shape):
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = next(model.parameters()).device
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        y = model(x)
    return {
        "parameters": params,
        "trainable_parameters": trainable,
        "input_shape": list(input_shape),
        "output_shape": list(y.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize model parameter counts and tensor shapes.")
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--shape", default="1,1,128,128,128")
    args = parser.parse_args()

    input_shape = tuple(int(v) for v in args.shape.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("unet3d", {}),
        ("dsa_unet3d_full", {"use_depthwise": True, "use_attention": True, "use_aspp": True}),
        ("dsa_no_depthwise", {"use_depthwise": False, "use_attention": True, "use_aspp": True}),
        ("dsa_no_attention", {"use_depthwise": True, "use_attention": False, "use_aspp": True}),
        ("dsa_no_aspp", {"use_depthwise": True, "use_attention": True, "use_aspp": False}),
    ]

    rows = []
    for name, kwargs in configs:
        model_name = "unet3d" if name == "unet3d" else "dsa_unet3d"
        model = build_model(model_name, base_channels=args.base_channels, **kwargs).to(device)
        summary = summarize_model(model, input_shape)
        summary["name"] = name
        summary["base_channels"] = args.base_channels
        rows.append(summary)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
