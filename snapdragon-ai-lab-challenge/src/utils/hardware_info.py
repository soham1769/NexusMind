import os
import platform
import sys
from typing import Dict, List, Any

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class SnapdragonHardwareDetector:
    """Detects Snapdragon hardware features, Qualcomm Hexagon NPU, and ONNX Execution Providers."""

    def __init__(self):
        self.os_name = platform.system()
        self.architecture = platform.machine()
        self.processor = platform.processor()

    def get_available_execution_providers(self) -> List[str]:
        """Returns ONNX Runtime available execution providers on the current host."""
        if ort is not None:
            try:
                return ort.get_available_providers()
            except Exception:
                pass
        return ["CPUExecutionProvider"]

    def is_snapdragon_hardware(self) -> bool:
        """Determines if the system is running on a Snapdragon ARM64 / Windows Copilot+ PC host."""
        is_arm64 = self.architecture.lower() in ["arm64", "aarch64"]
        proc_str = (self.processor or "").lower()
        env_str = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        
        is_qualcomm = any(kw in proc_str or kw in env_str for kw in ["snapdragon", "qualcomm", "sq3", "hexagon"])
        return is_arm64 or is_qualcomm

    def check_qnn_support(self) -> Dict[str, Any]:
        """Checks if Qualcomm Neural Processing (QNN) Execution Provider is available and active."""
        providers = self.get_available_execution_providers()
        has_qnn = "QNNExecutionProvider" in providers
        has_dml = "DmlExecutionProvider" in providers

        return {
            "has_qnn_provider": has_qnn,
            "has_directml_provider": has_dml,
            "recommended_provider": "QNNExecutionProvider" if has_qnn else ("DmlExecutionProvider" if has_dml else "CPUExecutionProvider"),
            "all_providers": providers
        }

    def get_hardware_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive system summary formatted for diagnostic logs."""
        qnn_info = self.check_qnn_support()
        return {
            "os": f"{self.os_name} {platform.release()} ({platform.version()})",
            "architecture": self.architecture,
            "processor": self.processor or "Qualcomm Snapdragon Compute Platform",
            "is_snapdragon": self.is_snapdragon_hardware(),
            "npu_acceleration_available": qnn_info["has_qnn_provider"] or qnn_info["has_directml_provider"],
            "primary_execution_provider": qnn_info["recommended_provider"],
            "installed_providers": qnn_info["all_providers"]
        }

if __name__ == "__main__":
    detector = SnapdragonHardwareDetector()
    summary = detector.get_hardware_summary()
    print("=== Snapdragon Hardware Detection ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
