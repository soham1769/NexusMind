import argparse
import sys
import time
from pathlib import Path

# Add project root directory to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
except ImportError:
    np = None

from src.models.qai_hub_loader import QualcommAIHubLoader
from src.models.npu_engine import NPUEngine
from src.pipeline.vision_processor import VisionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="SnapVision AI Benchmarking Utility")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--mock", action="store_true", default=True, help="Run benchmark in mock mode")
    return parser.parse_args()

def benchmark_provider(model_path: Path, provider_name: str, iterations: int, warmup: int):
    try:
        engine = NPUEngine(model_path=model_path, preferred_provider=provider_name)
        pipeline = VisionPipeline(engine=engine)
        
        # Warmup
        for _ in range(warmup):
            pipeline.process()

        # Measured runs
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            pipeline.process()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        if np is not None:
            latencies = np.array(latencies)
            mean_ms = float(np.mean(latencies))
            p95_ms = float(np.percentile(latencies, 95))
        else:
            mean_ms = sum(latencies) / len(latencies)
            p95_ms = mean_ms

        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "provider": engine.active_provider,
            "mean_ms": round(mean_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "fps": round(fps, 1),
            "status": "Success [OK]"
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "mean_ms": -1,
            "p95_ms": -1,
            "fps": -1,
            "status": f"Unavailable [X] ({e})"
        }

def print_benchmark_table(results, iterations):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=f"Performance Comparison ({iterations} Iterations)")
        table.add_column("Execution Provider", style="cyan")
        table.add_column("Mean Latency (ms)", style="bold magenta")
        table.add_column("95th Percentile Latency (ms)", style="magenta")
        table.add_column("Throughput (FPS)", style="bold green")
        table.add_column("Status", style="green")

        for r in results:
            table.add_row(
                r["provider"],
                str(r["mean_ms"]) if r["mean_ms"] > 0 else "N/A",
                str(r["p95_ms"]) if r["p95_ms"] > 0 else "N/A",
                str(r["fps"]) if r["fps"] > 0 else "N/A",
                r["status"]
            )
        console.print(table)
    except ImportError:
        print(f"\n--- Performance Comparison ({iterations} Iterations) ---")
        header = f"{'Execution Provider':<25} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'FPS':<8} | {'Status'}"
        print(header)
        print("-" * len(header))
        for r in results:
            mean_str = str(r["mean_ms"]) if r["mean_ms"] > 0 else "N/A"
            p95_str = str(r["p95_ms"]) if r["p95_ms"] > 0 else "N/A"
            fps_str = str(r["fps"]) if r["fps"] > 0 else "N/A"
            print(f"{r['provider']:<25} | {mean_str:<10} | {p95_str:<10} | {fps_str:<8} | {r['status']}")
        print("-" * len(header))

def main():
    args = parse_args()

    print("=== SnapVision AI Hardware Acceleration Benchmark ===\n")
    loader = QualcommAIHubLoader(output_dir="models")
    model_path = loader.download_and_compile_model(model_name="mobilenet_v4", mock=args.mock)

    providers_to_test = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    results = []

    for provider in providers_to_test:
        print(f"Benchmarking provider: {provider}...")
        res = benchmark_provider(model_path, provider, args.iterations, args.warmup)
        results.append(res)

    print_benchmark_table(results, args.iterations)
    print("\n* Note: QNNExecutionProvider offloads tensor computations directly to the Snapdragon Hexagon Tensor Processor (HTP).")

if __name__ == "__main__":
    main()
