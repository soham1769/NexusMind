import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

try:
    import numpy as np
except ImportError:
    np = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.utils.logger import setup_logger
from src.utils.hardware_info import SnapdragonHardwareDetector

logger = setup_logger("NPUEngine")

class NPUEngine:
    """ONNX Runtime inference engine configured for Qualcomm Snapdragon Hexagon NPU offloading."""

    def __init__(self, model_path: Path, preferred_provider: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.hardware_detector = SnapdragonHardwareDetector()
        self.preferred_provider = preferred_provider

        if ort is not None:
            self.session, self.active_provider = self._create_inference_session()
            self.input_meta = self.session.get_inputs()[0]
            self.output_meta = self.session.get_outputs()[0]
        else:
            logger.warning("[NPUEngine] 'onnxruntime' module not found. Running in mock engine mode.")
            self.session = None
            self.active_provider = self.preferred_provider or "QNNExecutionProvider (Mock)"
            self.input_meta = type('Meta', (), {'name': 'input', 'shape': [1, 3, 224, 224]})()
            self.output_meta = type('Meta', (), {'name': 'output', 'shape': [1, 1000]})()

    def _build_provider_options(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Configures ONNX Runtime provider options for Snapdragon QNN NPU & DirectML."""
        if ort is None:
            return [("CPUExecutionProvider", {})]

        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Execution Providers on system: {available_providers}")

        provider_list = []

        if "QNNExecutionProvider" in available_providers:
            qnn_options = {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "profiling_level": "basic",
                "enable_htp_fp16_precision": "1"
            }
            provider_list.append(("QNNExecutionProvider", qnn_options))

        if "DmlExecutionProvider" in available_providers:
            dml_options = {"device_id": 0}
            provider_list.append(("DmlExecutionProvider", dml_options))

        provider_list.append(("CPUExecutionProvider", {}))

        if self.preferred_provider:
            matched = [p for p in provider_list if p[0] == self.preferred_provider]
            if matched:
                return matched + [("CPUExecutionProvider", {})]

        return provider_list

    def _create_inference_session(self) -> Tuple[Any, str]:
        """Initializes the ONNX Runtime session with hardware acceleration."""
        providers = self._build_provider_options()

        logger.info(f"Initializing InferenceSession for {self.model_path.name}...")
        try:
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            active_provider = session.get_providers()[0]
            
            if "QNN" in active_provider:
                logger.info("[SUCCESS] Hardware Acceleration ACTIVE: Offloaded to Snapdragon Hexagon NPU via QNN Execution Provider!")
            elif "Dml" in active_provider:
                logger.info("[DIRECTML] Hardware Acceleration ACTIVE: DirectML Provider selected.")
            else:
                logger.warning("[INFO] Running on CPU Execution Provider (NPU/GPU drivers not detected).")

            return session, active_provider
        except Exception as e:
            logger.error(f"Failed to create accelerated session: {e}. Falling back to default CPU provider.")
            try:
                session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                return session, "CPUExecutionProvider"
            except Exception:
                return None, "CPUExecutionProvider (Mock)"

    def run_inference(self, input_tensor: Any) -> Tuple[Any, float]:
        """Executes hardware-accelerated inference and returns (output_tensor, latency_ms)."""
        start_time = time.perf_counter()

        if self.session is not None and np is not None:
            input_name = self.input_meta.name
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)

            outputs = self.session.run(None, {input_name: input_tensor})
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            return outputs[0], latency_ms

        # Mock inference simulation (sub-5ms NPU emulation)
        time.sleep(0.0042)  # 4.2ms simulation
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        if np is not None:
            mock_output = np.random.randn(1, 1000).astype(np.float32)
        else:
            mock_output = [[0.0] * 1000]

        return mock_output, latency_ms

    def get_session_info(self) -> Dict[str, Any]:
        """Returns metadata regarding active providers and tensor shapes."""
        return {
            "model_path": str(self.model_path),
            "active_provider": self.active_provider,
            "input_name": self.input_meta.name,
            "input_shape": getattr(self.input_meta, 'shape', [1, 3, 224, 224]),
            "output_name": self.output_meta.name,
            "output_shape": getattr(self.output_meta, 'shape', [1, 1000])
        }
