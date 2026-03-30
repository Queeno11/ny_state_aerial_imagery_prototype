import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. MODEL REGISTRY SETUP
# ==========================================
_MODEL_REGISTRY = {}

def register_model(name):
    """
    Decorator to automatically register models into the dictionary.
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name, **kwargs):
    """
    Factory function to instantiate a model by its string name.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)


# ==========================================
# 2. MINIMAL TEST MODEL
# ==========================================
@register_model("small_cnn")
class SmallCNN(nn.Module):
    def __init__(self, resizing_size, bands=4, kind="reg"):
        """
        A minimal CNN to test the PyTorch transition.
        """
        super().__init__()
        self.kind = kind
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=bands, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate the size of the tensor after 2 max-pooling layers (halved twice)
        # e.g., 224 -> 112 -> 56
        flattened_dim = resizing_size // 4
        linear_input_size = 32 * flattened_dim * flattened_dim
        
        # Fully Connected Block
        self.fc1 = nn.Linear(linear_input_size, 64)
        
        # Output layer (1 neuron for regression, N neurons for classification)
        out_features = 1 if kind == "reg" else 2 # Assuming binary classification for testing
        self.fc2 = nn.Linear(64, out_features)

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        
        # Layer 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Layer 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for Dense layers (keep batch dimension)
        x = torch.flatten(x, 1)
        
        # Dense -> ReLU
        x = F.relu(self.fc1(x))
        
        # Output
        x = self.fc2(x)
        
        # Note: In PyTorch, BCEWithLogitsLoss or CrossEntropyLoss handles the Softmax/Sigmoid natively.
        # So we return the raw logits here.
        return x