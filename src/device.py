import platform

def get_device():
    """Detects available hardware and returns the appropriate PyTorch device."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using device: cuda")
            print(f"GPU: {gpu_name}")
            return torch.device("cuda")
        else:
            print("Using device: cpu")
            return torch.device("cpu")
    except ImportError:
        print("PyTorch not installed. Falling back to cpu string.")
        return "cpu"

def print_environment_diagnostics():
    """Prints detailed system and PyTorch environment diagnostics."""
    print("=" * 50)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)
    print(f"Python version: {platform.python_version()}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        try:
            import torchvision
            print(f"torchvision version: {torchvision.__version__}")
        except ImportError:
            print("torchvision not installed.")
            
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA available: {cuda_avail}")
        
        if cuda_avail:
            print(f"CUDA version: {torch.version.cuda}")
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch is not installed.")
        
    print(f"CPU info: {platform.processor()}")
    print("=" * 50)
