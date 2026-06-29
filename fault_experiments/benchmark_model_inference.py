import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fault_experiments.infer_real_volume import load_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "U-Net": ROOT / "checkpoints/unet3d_thebe_e8.pt",
    "Hybrid DSA": ROOT / "checkpoints/hybrid_dsa_thebe_e8.pt",
    "SwinUNETR": ROOT / "checkpoints/swinunetr_f3chain_thebe_e8.pt",
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
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


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
