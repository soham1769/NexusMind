import argparse
import sys
from pathlib import Path

# Add project root directory to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.hardware_info import SnapdragonHardwareDetector
from src.utils.logger import setup_logger
from src.models.qai_hub_loader import QualcommAIHubLoader
from src.models.npu_engine import NPUEngine
from src.pipeline.vision_processor import VisionPipeline

logger = setup_logger("SnapVisionCLI")

def parse_args():
    parser = argparse.ArgumentParser(
        description="SnapVision AI: On-Device Snapdragon NPU Accelerated Vision Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v4",
        help="Qualcomm AI Hub model to compile & run (default: mobilenet_v4)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock compilation/model loader mode for demonstration"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        help="Force specific ONNX Execution Provider"
    )
    return parser.parse_args()

def print_header():
    title = "========================================================\n" \
            "                  SnapVision AI Core                   \n" \
            "    On-Device Snapdragon Hexagon NPU Accelerated Suite \n" \
            "    Qualcomm AI Hub Submission for HP Copilot+ PCs     \n" \
            "========================================================"
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(
            Panel.fit(
                "[bold cyan]SnapVision AI[/bold cyan]\n"
                "[bright_white]On-Device Snapdragon Hexagon NPU Accelerated Suite[/bright_white]\n"
                "[dim]Qualcomm AI Hub Challenge Submission for HP Copilot+ PCs[/dim]",
                border_style="cyan"
            )
        )
    except ImportError:
        print(title)

def print_table(hw_info):
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="System Hardware & Hardware Acceleration Status")
        table.add_column("Property", style="bold green")
        table.add_column("Detected Value", style="magenta")

        table.add_row("Operating System", hw_info["os"])
        table.add_row("Architecture", hw_info["architecture"])
        table.add_row("Processor Platform", hw_info["processor"])
        table.add_row("Is Snapdragon Device?", "Yes [OK]" if hw_info["is_snapdragon"] else "No (Fallback Mode)")
        table.add_row("NPU Acceleration Status", "Active [NPU]" if hw_info["npu_acceleration_available"] else "CPU Fallback")
        table.add_row("Primary Provider", hw_info["primary_execution_provider"])
        console.print(table)
    except ImportError:
        print("\n--- System Hardware & Hardware Acceleration Status ---")
        for k, v in hw_info.items():
            print(f"  {k:<30}: {v}")
        print("----------------------------------------------------\n")

def main():
    args = parse_args()
    print_header()

    # 1. Detect Snapdragon Hardware
    detector = SnapdragonHardwareDetector()
    hw_info = detector.get_hardware_summary()
    print_table(hw_info)

    # 2. Fetch / Compile Model from Qualcomm AI Hub
    print("\n[Step 1: Qualcomm AI Hub Model Access & Compilation]")
    loader = QualcommAIHubLoader(output_dir="models")
    model_path = loader.download_and_compile_model(
        model_name=args.model,
        mock=args.mock
    )
    print(f"Model artifact ready at: {model_path}")

    # 3. Instantiate NPU Inference Engine
    print("\n[Step 2: Initializing NPU Engine & Execution Provider]")
    engine = NPUEngine(model_path=model_path, preferred_provider=args.provider)
    session_info = engine.get_session_info()
    print(f"Active Provider: {session_info['active_provider']}")

    # 4. Execute Vision Pipeline
    print("\n[Step 3: Running Vision Inference Pipeline]")
    pipeline = VisionPipeline(engine=engine)
    result = pipeline.process(image_input=None)

    print(f"Inference Latency: {result['latency_ms']} ms")
    print(f"Estimated Throughput: {result['fps']} FPS")
    print("\nTop Predictions (Class Index, Confidence):")
    for idx, conf in result["top_predictions"]:
        print(f"  - Class #{idx:04d}: {conf * 100:.2f}%")

    print("\n[SUCCESS] SnapVision AI pipeline completed successfully!")

if __name__ == "__main__":
    main()
