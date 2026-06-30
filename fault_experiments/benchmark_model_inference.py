import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fault_experiments.infer_real_volume import load_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "U-Net": ROOT / "runs/thebe_adaptation/unet3d_e8/best.pt",
    "Hybrid DSA": ROOT / "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt",
    "SwinUNETR": ROOT / "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt",
}


def benchmark(model, tensor, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(tensor)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    return {
        "median_ms": float(np.median(timings)),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings, ddof=1)),
        "throughput_patches_per_second": float(1000.0 / np.median(timings)),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def module_macs_lower_bound(model, tensor):
    total = 0
    handles = []

    def convolution_hook(module, inputs, output):
        nonlocal total
        kernel = int(np.prod(module.kernel_size))
        macs_per_output = module.in_channels // module.groups * kernel
        total += int(output.numel()) * macs_per_output

    def linear_hook(module, inputs, output):
        nonlocal total
        total += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(tensor)
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    return total


class CudaGraphWrapper:
    def __init__(self, model, example):
        self.model = model
        self.static_input = example.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                model(self.static_input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_output = model(self.static_input)

    def __call__(self, tensor):
        self.static_input.copy_(tensor)
        self.graph.replay()
        return self.static_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="U-Net,Hybrid DSA,SwinUNETR")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/efficiency/inference_benchmark.json"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    torch.manual_seed(20260627)
    tensor = torch.randn(1, 1, 128, 128, 128, device=device)
    results = []
    for name in [value.strip() for value in args.models.split(",") if value.strip()]:
        torch.cuda.empty_cache()
        model, _ = load_model(CHECKPOINTS[name], device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        macs_lower_bound = module_macs_lower_bound(model, tensor)
        execution = "eager"
        if args.compile:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            execution = "torch_compile_reduce_overhead"
        elif args.cuda_graph:
            model = CudaGraphWrapper(model, tensor)
            execution = "cuda_graph"
        metrics = benchmark(model, tensor, args.warmup, args.repeats)
        row = {
            "model": name,
            "execution": execution,
            "parameter_count": parameter_count,
            "patch_shape": [128, 128, 128],
            "dtype": "float32",
            "module_macs_lower_bound_per_patch": macs_lower_bound,
            "module_gmacs_lower_bound_per_patch": macs_lower_bound / 1e9,
            "mac_note": "Counts Conv3d, ConvTranspose3d and Linear multiply-accumulates. It excludes normalization, activation, interpolation and explicit attention matrix products, so it is a lower bound, especially for SwinUNETR.",
            **metrics,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
    output = args.output
    if args.compile:
        output = output.with_name(output.stem + "_compiled" + output.suffix)
    elif args.cuda_graph:
        output = output.with_name(output.stem + "_cuda_graph" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
