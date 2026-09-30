import torch.nn as nn
try:
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:
    pass

def create_model(num_classes: int, pretrained: bool = True):
    """Creates an EfficientNet-B0 model replacing the classifier for num_classes."""
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)
        
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
