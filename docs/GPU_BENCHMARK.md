# GPU Benchmarking Methodology

## Why GPU Acceleration?
Deep learning models, especially Convolutional Neural Networks (CNNs) like EfficientNet-B0, require billions of mathematical operations (matrix multiplications) to classify a single image. A standard CPU handles these sequentially across a few cores, whereas an NVIDIA GPU can distribute the load across thousands of parallel CUDA cores.

## How PyTorch Uses CUDA
PyTorch provides a seamless interface to CUDA via `torch.device('cuda')`. By transferring both the model parameters (`model.to(device)`) and the input image tensors (`inputs.to(device)`) to the GPU VRAM, PyTorch bypasses the CPU and executes the forward pass natively on the graphics card.

## The Importance of Batch Processing
Processing images one by one (batch size = 1) is highly inefficient on a GPU because it leaves most of the parallel cores idle. By grouping images into batches (e.g., 32 or 64 images at once), we can fully saturate the CUDA cores. This dramatically increases overall **throughput** (images processed per second) even though the latency of an individual batch increases slightly.

### Hardware
- **GPU Model:** (To be dynamically extracted and reported from `nvidia-smi` inside the Colab environment).
- **VRAM Capacity:** (Extracted via `torch.cuda.get_device_properties()`).

### Software
- **Python:** 3.10+
- **PyTorch:** Latest stable CUDA-compatible build.
- **CUDA Toolkit:** Matches the target runtime driver.

### Workload
- **Model:** Pretrained `EfficientNet-B0`.
- **Image Resolution:** 224x224 RGB.
- **Dataset Phase:** Only the validation subset is used during benchmarking to guarantee fair comparisons.
- **Batch Sizes Tested:** 8, 16, 32, 64.

### Metrics
1. **Training Time:** The wall-clock duration of optimization loops per epoch.
2. **Inference Latency:** Average time required to process a single batch of images.
3. **Throughput:** Total number of images processed per second globally.
4. **GPU Speedup Factor:** Calculated mathematically as `CPU_Time / GPU_Time`.

### Methodology: Synchronization and Warm-ups
Benchmarking PyTorch on GPUs is notoriously tricky due to asynchronous execution. When the CPU dispatches a command to the GPU, it immediately moves to the next line of code without waiting for the GPU to finish. 
To measure actual computation time, we must use `torch.cuda.Event` and call `torch.cuda.synchronize()`. This creates a hard barrier, forcing the CPU to wait until the GPU has fully completed the batch before calculating the elapsed time.
Additionally, we perform explicitly untimed "warm-up" iterations before opening the performance timers to bypass the initial overhead of CUDA context initialization and PyTorch caching logic.
