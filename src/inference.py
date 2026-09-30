import time
try:
    import torch
except ImportError:
    pass
from src.device import get_device

def run_batch_inference(model, dataloader, device):
    model = model.to(device)
    model.eval()
    
    total_images = 0
    start_time = time.time()
    predictions = []
    
    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            for i in range(inputs.size(0)):
                predictions.append({
                    'path': paths[i],
                    'class_idx': preds[i].item(),
                    'confidence': confs[i].item()
                })
            total_images += inputs.size(0)
            
    total_time = time.time() - start_time
    throughput = total_images / total_time if total_time > 0 else 0
    
    return predictions, total_time, throughput
