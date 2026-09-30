# GPU-Accelerated Plant Disease Detection

## Project Overview
This project provides an industry-quality, PyTorch-based capstone demonstrating the architectural integration and performance benefits of NVIDIA GPU acceleration for deep learning image classification.

## Problem Statement
Plant diseases reduce agricultural yield and threaten food security. While CNNs offer high accuracy in disease classification, deploying these systems at scale requires massive computational resources. Understanding the performance characteristics of these models across CPU and GPU hardware is essential.

## Objectives
- Develop an image classification system for plant diseases using EfficientNet-B0.
- Provide dynamic support for both CPU and NVIDIA CUDA execution.
- Implement a scientifically sound benchmarking suite to measure latency and throughput.
- Compare CPU vs. GPU performance scaling across batch sizes.

## GPU Computing Approach
GPU acceleration requires explicit architectural integration:
1. **Model Placement:** Network weights are transferred to GPU VRAM using `model.to(device)`.
2. **Tensor Placement:** Incoming image batches are pushed to the GPU (`inputs.to(device)`).
3. **Synchronization:** Accurate timing requires `torch.cuda.Event` and `torch.cuda.synchronize()` to account for PyTorch's asynchronous execution.

## Architecture
`Dataset` -> `Preprocessing (torchvision)` -> `DataLoader` -> `EfficientNet-B0` -> `GPU Training` -> `GPU Batch Inference` -> `Benchmarking` -> `Results (CSV/Plots)`

## Technology Stack
- **Python 3.13**
- **PyTorch** & **torchvision**
- **NumPy**, **Pandas**, **Matplotlib**
- **scikit-learn**

## Project Structure
```text
src/          # Core model, train, inference, and benchmark logic
scripts/      # CLI execution wrappers
data/         # Datasets (raw & processed)
models/       # Saved PyTorch checkpoint weights (.pth)
results/      # Output CSVs and comparative charts
notebooks/    # Google Colab execution files
tests/        # Unit tests
docs/         # Architecture and methodology notes
```

## Installation
Ensure you have Python 3.13 installed.
```bash
pip install -r requirements.txt
```

## CPU Development
This project is fully compatible with CPU-only machines for development and testing. The device configuration will gracefully fallback to CPU if CUDA is unavailable.

## CUDA/GPU Setup
To leverage GPU acceleration, you must have an NVIDIA GPU and the appropriate CUDA toolkit installed alongside a CUDA-compatible version of PyTorch (from `pytorch.org`).

## Phase 2 - Dataset Preparation
1. Download/obtain a plant disease dataset (e.g., PlantVillage).
2. Place it under `data/raw/`.
3. Ensure ImageFolder directory structure (where subfolder names map to class names).
4. Verify the dataset format and counts by running:
   ```bash
   python scripts/prepare_dataset.py
   ```
5. Perform a dry-run of the training pipeline (tests parsing, classes, and model creation without executing epochs):
   ```bash
   python scripts/train_model.py --dry-run
   ```

## GPU Training Experiment
Due to massive hardware constraints on local CPUs, actual training and benchmarking are designed to be executed via Google Colab.
- **Why Google Colab?** It provides free, ephemeral access to high-performance NVIDIA GPUs (like the T4 or V100), avoiding the need to purchase dedicated hardware.
- **GPU Verification:** The notebook explicitly validates `!nvidia-smi` and `torch.cuda.is_available()` before executing to guarantee authentic hardware metrics.
- **Dataset Setup:** The dataset is mounted securely to the cloud environment, preventing massive Git repository bloating.
- **EfficientNet-B0:** We utilize transfer learning, keeping the backbone static while training a dynamic classification head.
- **GPU Training & CPU Baseline:** We run heavy loops on the GPU and then lock a subsequent loop strictly to the CPU to measure the baseline.
- **GPU Benchmark Methodology:** Accurate scaling throughput across exponentially increasing batch sizes, measuring the time delta using PyTorch CUDA Events.
- **CUDA Synchronization & Warmups:** By enforcing `torch.cuda.synchronize()` and executing warm-up loops, we eliminate asynchronous dispatch delays and caching overhead from our measurements.

## Benchmarking
To compare execution performance:
```bash
python scripts/benchmark_cpu_gpu.py
```

## Results
*[Placeholder: Quantitative speedup factors, throughput graphs, and latency tables will be documented here upon execution in the Google Colab target environment.]*

## Reproducibility
Random seeds are strictly enforced within `src/config.py` to ensure reproducible data splits and training regimes.

## Future Improvements
- Expand model architecture choices (e.g., ResNet, MobileNet).
- Build a REST API or web interface for real-time demonstration.
- Deploy via TensorRT or ONNX for optimized edge execution.
