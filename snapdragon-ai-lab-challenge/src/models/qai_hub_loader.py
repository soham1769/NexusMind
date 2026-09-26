import os
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("QAIHubLoader")

class QualcommAIHubLoader:
    """Loads, compiles, and optimizes models using the Qualcomm AI Hub Python SDK (qai_hub)."""

    def __init__(
        self,
        target_device_name: str = "Snapdragon X Elite CRD",
        output_dir: str = "models",
        api_token: Optional[str] = None
    ):
        self.target_device_name = target_device_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_token = api_token or os.environ.get("QAI_HUB_API_TOKEN")

    def download_and_compile_model(
        self,
        model_name: str = "mobilenet_v4",
        target_runtime: str = "qnn_context_binary",
        mock: bool = False
    ) -> Path:
        """Downloads a pre-trained model from Qualcomm AI Hub and compiles it for Hexagon NPU."""
        target_filename = self.output_dir / f"{model_name}_snapdragon_npu.onnx"

        if mock or not self.api_token:
            logger.warning(
                "[QAI Hub] API token not provided or mock mode enabled. "
                "Generating placeholder target model artifact for demonstration."
            )
            return self._generate_mock_onnx_model(target_filename)

        try:
            import qai_hub as hub

            logger.info(f"[QAI Hub] Initializing session for model: '{model_name}'...")
            hub_model = hub.get_model(model_name)

            devices = hub.get_devices(name=self.target_device_name)
            if not devices:
                logger.warning(f"Target device '{self.target_device_name}' not found. Fallback to Snapdragon device search.")
                devices = hub.get_devices(attributes="chipset:snapdragon-x-elite")
            
            target_device = devices[0]
            logger.info(f"[QAI Hub] Selected compilation target: {target_device.name}")

            logger.info(f"[QAI Hub] Submitting compilation job targeting QNN Execution Engine...")
            compile_job = hub.submit_compile_job(
                model=hub_model,
                device=target_device,
                options="--target_runtime qnn_context_binary --quantize_io"
            )

            logger.info(f"[QAI Hub] Compilation job submitted. Job ID: {compile_job.job_id}")
            compiled_model = compile_job.get_target_model()

            logger.info(f"[QAI Hub] Downloading optimized NPU binary to {target_filename}...")
            compiled_model.download(str(target_filename))
            logger.info("[QAI Hub] Download complete successfully!")

            return target_filename

        except Exception as e:
            logger.error(f"[QAI Hub] Remote compilation failed or SDK not configured: {e}")
            logger.info("[QAI Hub] Falling back to local ONNX model stub.")
            return self._generate_mock_onnx_model(target_filename)

    def _generate_mock_onnx_model(self, target_path: Path) -> Path:
        """Creates a placeholder model file for testing hardware execution pipelines."""
        if target_path.exists():
            return target_path

        try:
            import numpy as np
            import onnx
            from onnx import helper, TensorProto

            X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
            Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])

            W_val = np.random.randn(3 * 224 * 224, 1000).astype(np.float32) * 0.01
            W = helper.make_tensor('W', TensorProto.FLOAT, [3 * 224 * 224, 1000], W_val.flatten().tolist())

            flatten_node = helper.make_node('Flatten', ['input'], ['flattened'], axis=1)
            gemm_node = helper.make_node('Gemm', ['flattened', 'W'], ['output'], alpha=1.0, beta=1.0)

            graph_def = helper.make_graph(
                [flatten_node, gemm_node],
                'snapdragon_demo_graph',
                [X],
                [Y],
                initializer=[W]
            )

            model_def = helper.make_model(graph_def, producer_name='snapvision-qai-hub-stub')
            onnx.save(model_def, str(target_path))
            logger.info(f"[Stub Generator] Valid sample ONNX model saved to: {target_path}")
        except Exception:
            # Fallback placeholder file
            with open(target_path, "wb") as f:
                f.write(b"SNAPDRAGON_QNN_MOCK_MODEL_BINARY_STUB_QUALCOMM_AI_HUB")
            logger.info(f"[Stub Generator] Mock model binary saved to: {target_path}")

        return target_path
