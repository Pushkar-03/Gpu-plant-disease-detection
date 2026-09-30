import time
try:
    import torch
except ImportError:
    pass

def run_benchmark(model, dataloader, device):
    model = model.to(device)
    model.eval()
    
    print(f"Starting benchmark on {device}...")
    print("Performing warm-up iterations...")
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            _ = model(inputs)
            if i >= 1: 
                break
                
    if hasattr(device, 'type') and device.type == 'cuda':
        torch.cuda.synchronize()
        
    total_time = 0.0
    num_images = 0
    
    print("Measuring performance...")
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            if hasattr(device, 'type') and device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                _ = model(inputs)
                
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                t0 = time.perf_counter()
                _ = model(inputs)
                batch_time = time.perf_counter() - t0
                
            total_time += batch_time
            num_images += inputs.size(0)
            
    throughput = num_images / total_time if total_time > 0 else 0
    avg_batch_time = total_time / len(dataloader) if len(dataloader) > 0 else 0
    
    print(f"Benchmark Complete on {device}")
    print(f"Total Images: {num_images} | Throughput: {throughput:.2f} img/sec")
    
    return {
        'device': str(device),
        'num_images': num_images,
        'total_time_seconds': total_time,
        'images_per_second': throughput,
        'avg_batch_time': avg_batch_time
    }
