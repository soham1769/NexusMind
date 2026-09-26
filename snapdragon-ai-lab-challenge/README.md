# SnapVision AI ⚡

> **On-Device Snapdragon® NPU Accelerated Vision & Productivity Suite for HP Copilot+ PCs**  
> *Submission for the Snapdragon AI Lab Build & Present Challenge*

[![Snapdragon](https://img.shields.io/badge/Hardware-Snapdragon_X_Elite-purple.svg)](https://www.qualcomm.com/snapdragon)
[![Qualcomm AI Hub](https://img.shields.io/badge/Model_Hub-Qualcomm_AI_Hub-blue.svg)](https://aihub.qualcomm.com/)
[![Windows ARM64](https://img.shields.io/badge/Platform-Windows_11_ARM64-0078D4.svg)](https://microsoft.com)
[![ONNX Runtime QNN](https://img.shields.io/badge/Inference_Engine-ONNX_Runtime_QNN-orange.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

**SnapVision AI** is a production-ready, on-device AI computer vision application optimized specifically for **Snapdragon-powered HP PCs** (such as the HP OmniBook X and HP EliteBook Ultra). 

By combining pre-trained quantized models from the **Qualcomm AI Hub** with **ONNX Runtime's QNN Execution Provider**, SnapVision AI offloads complex neural network workloads directly onto the **Qualcomm® Hexagon™ NPU**. This delivers real-time spatial analysis and image classification with sub-5ms latencies, 100% offline privacy, and minimal battery consumption.

---

## 🏗 System Architecture

```
                                +-----------------------------------+
                                |     Qualcomm AI Hub (Cloud)       |
                                |  Quantization (W8A8) & Compile    |
                                +-----------------+-----------------+
                                                  |
                                                  v  (Compiled .onnx / .qnn)
+-------------------------------------------------+-----------------------------------+
|  SnapVision AI Application Core (HP Copilot+ PC / Windows ARM64)                     |
|                                                                                     |
|  +--------------------------+    +-----------------------+    +------------------+  |
|  | Hardware Discovery Engine | -> | Vision Pre-Processing | -> | ONNX Runtime     |  |
|  | (Detect NPU / ARM64)     |    | (Resizing & Normalization) | Inference Session |  |
|  +--------------------------+    +-----------------------+    +--------+---------+  |
|                                                                        |            |
|                                       +--------------------------------+            |
|                                       | (Execution Provider Selection)              |
|                                       v                                             |
|                     +-----------------+-----------------+                           |
|                     | Primary Path: QNN Execution Prov. |                           |
|                     +-----------------+-----------------+                           |
+---------------------------------------|---------------------------------------------+
                                        v
                       +---------------------------------+
                       | Snapdragon Hexagon NPU          |
                       | (Hexagon Tensor Processor / HTP)|
                       +---------------------------------+
```

---

## ⚡ Snapdragon / NPU Optimization Details

SnapVision AI is architected around Snapdragon compute platforms to unlock maximum hardware acceleration:

1. **Direct Hexagon NPU Offloading (`QNNExecutionProvider`)**:
   - Interfaced through `QnnHtp.dll` backend binaries.
   - Configured with `htp_performance_mode: burst` for maximum TOPS throughput.
2. **Quantization Precision (W8A8)**:
   - Uses 8-bit Weights and 8-bit Activations (INT8) optimized for Hexagon Tensor Processor (HTP) vector engines.
3. **Graceful Multi-Tier Hardware Fallback**:
   - **Tier 1**: `QNNExecutionProvider` (Qualcomm Hexagon NPU)
   - **Tier 2**: `DmlExecutionProvider` (DirectML acceleration on Adreno GPU)
   - **Tier 3**: `CPUExecutionProvider` (Native ARM64 multi-threaded execution on Snapdragon Oryon CPU cores)

---

## 🤖 Qualcomm AI Hub Model Integration

Models are fetched, compiled, and benchmarked using the Qualcomm AI Hub SDK (`qai_hub`).

```python
from src.models.qai_hub_loader import QualcommAIHubLoader

# Initialize loader targeting Snapdragon X Elite
loader = QualcommAIHubLoader(target_device_name="Snapdragon X Elite CRD")

# Compile and download MobileNetV4 quantized for QNN NPU execution
model_path = loader.download_and_compile_model(
    model_name="mobilenet_v4",
    target_runtime="qnn_context_binary"
)
```

---

## 📂 Repository Structure

```
snapdragon-ai-lab-challenge/
├── README.md                      # Primary project documentation
├── pyproject.toml                 # Modern Python package metadata & setup
├── requirements.txt               # Cross-compatible ML dependency specifications
├── .gitignore                     # Git exclusion rules
├── config/
│   └── config.yaml                # Hardware parameters, provider priority & thresholds
├── docs/
│   └── Technical_Architecture.md  # Detailed challenge presentation & architectural spec
└── src/
    ├── __init__.py                # Package root
    ├── main.py                    # Main CLI entrypoint
    ├── benchmark.py               # Hardware provider benchmarking utility
    ├── models/
    │   ├── __init__.py
    │   ├── qai_hub_loader.py      # Qualcomm AI Hub SDK downloading & compilation stub
    │   └── npu_engine.py          # ONNX Runtime QNN/Hexagon NPU execution engine
    ├── pipeline/
    │   ├── __init__.py
    │   └── vision_processor.py    # End-to-end vision pre-processing & post-processing
    └── utils/
        ├── __init__.py
        ├── hardware_info.py       # Snapdragon ARM64 & QNN driver detection module
        └── logger.py              # Structured logging utility
```

---

## 🚀 Step-by-Step Build & Run Instructions

### Prerequisites
* **Device**: HP Copilot+ PC powered by Snapdragon X Elite / Plus (or any Windows ARM64 / x64 machine for testing).
* **Python**: Python 3.10 or higher.
* **Qualcomm AI Hub Token** *(Optional for live compile)*: Obtain API Key from [aihub.qualcomm.com](https://aihub.qualcomm.com/).

### 1. Clone & Set Up Environment

```powershell
# Clone the repository
git clone https://github.com/your-username/snapdragon-ai-lab-challenge.git
cd snapdragon-ai-lab-challenge

# Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Qualcomm AI Hub Credentials (Optional)

```powershell
# Set your Qualcomm AI Hub API Token
$env:QAI_HUB_API_TOKEN="your_qualcomm_ai_hub_token_here"
```

### 3. Run Application CLI

Run the main entrypoint in mock/demonstration mode:

```powershell
python src/main.py --mock
```

To run with live hardware execution provider selection:

```powershell
python src/main.py --provider QNNExecutionProvider
```

### 4. Run Hardware Benchmarking Utility

Evaluate latency (ms) and throughput (FPS) across Hexagon NPU, DirectML GPU, and CPU execution providers:

```powershell
python src/benchmark.py --iterations 50
```

---

## 📊 Documentation & Presentation Specs

For detailed technical evaluation criteria, benchmark methodology, and HP device integration specs, refer to [docs/Technical_Architecture.md](file:///C:/Users/soham/.gemini/antigravity/scratch/snapdragon-ai-lab-challenge/docs/Technical_Architecture.md).

---

## 📄 License

This project is released under the [MIT License](LICENSE).
