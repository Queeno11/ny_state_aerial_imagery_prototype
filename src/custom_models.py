import importlib.util
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

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

# ==========================================
# 3. SCALEMAE MODEL
# ==========================================

def _load_scalemae_backbone(local_dir="./pretrained/scalemae"):
    """
    Downloads model.py from the MVRL HuggingFace repo, imports it dynamically,
    and returns the backbone with pretrained weights via from_pretrained().
    """
    # Step 1: Download the model definition (not a .pth — the repo ships its own class)
    model_file = hf_hub_download(
        repo_id="MVRL/scalemae-vitlarge-800",
        filename="model.py",
        local_dir=local_dir
    )

    # Step 2: Dynamically import the downloaded model.py as a module
    spec = importlib.util.spec_from_file_location("scalemae_module", model_file)
    scalemae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scalemae_module)

    # Step 3: Load backbone + pretrained weights via from_pretrained
    # This handles weight downloading internally — no torch.load needed
    backbone = scalemae_module.ScaleMAE_baseline.from_pretrained("MVRL/scalemae-vitlarge-800")

    return backbone


@register_model("scalemae")
class ScaleMAE(nn.Module):
    def __init__(self, resizing_size=224, bands=3, kind="reg", freeze_strategy="linear_probe"):
        super().__init__()
        self.kind = kind
        self.patch_size = 16  # Fixed for ViT-Large/16; passed at runtime to backbone

        # 1. Load backbone with pretrained weights from HuggingFace
        print("📥 Downloading/Loading Official ScaleMAE weights from Hugging Face Hub...")
        try:
            self.backbone = _load_scalemae_backbone()
            print("✅ ScaleMAE weights loaded successfully via from_pretrained!")
        except Exception as e:
            print(f"⚠️ Failed to load ScaleMAE weights. Training from scratch. Error: {e}")
            # Fallback: plain timm ViT-L without ScaleMAE pretraining
            self.backbone = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                in_chans=bands,
                num_classes=0
            )

        # 2. Regression/classification head
        # ScaleMAE_baseline.from_pretrained returns patch tokens of shape (B, N, embed_dim)
        # ViT-Large embed_dim = 1024
        embed_dim = 1024
        out_features = 1 if kind == "reg" else 2

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),                  # GELU matches ViT internals (was ReLU before — fixed)
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, out_features)
        )

        # 3. Apply freezing strategy
        self._apply_freezing(freeze_strategy)

    def _apply_freezing(self, strategy):
        if strategy == "none":
            print("🔥 ScaleMAE fully unfrozen — all parameters trainable.")
            return

        vit = self.backbone.model  # ← the actual VisionTransformer lives here

        # Freeze embedding layers
        for param in vit.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(vit, 'cls_token'):
            vit.cls_token.requires_grad = False
        if hasattr(vit, 'pos_embed'):
            vit.pos_embed.requires_grad = False

        num_blocks = len(vit.blocks)  # 24 for ViT-Large

        if strategy == "partial":
            blocks_to_freeze = int(num_blocks * 0.75)   # freeze blocks 0–17, train 18–23
        elif strategy == "linear_probe":
            blocks_to_freeze = num_blocks               # freeze all, head only
        else:
            blocks_to_freeze = 0

        for i in range(blocks_to_freeze):
            for param in vit.blocks[i].parameters():
                param.requires_grad = False

        # Also freeze the final LayerNorm if doing linear probe
        if strategy == "linear_probe":
            for param in vit.norm.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        print(f"❄️ ScaleMAE '{strategy}': froze embeddings + {blocks_to_freeze}/{num_blocks} blocks "
            f"| Trainable: {trainable/1e6:.1f}M | Frozen: {frozen/1e6:.1f}M params")
    
    def unfreeze_stage(self, stage: int):
        """
        Progressive unfreezing for staged fine-tuning.
        Stage 0: head only (default at init with linear_probe)
        Stage 1: unfreeze last 25% of blocks (blocks 18-23 for ViT-L/24)
        Stage 2: unfreeze next 25% of blocks (blocks 12-17)
        """
        vit = self.backbone.model
        num_blocks = len(vit.blocks)  # 24

        if stage == 1:
            # Unfreeze last 25% — blocks 18-23
            unfreeze_from = int(num_blocks * 0.75)  # block 18
            for i in range(unfreeze_from, num_blocks):
                for param in vit.blocks[i].parameters():
                    param.requires_grad = True
            # Also unfreeze the final LayerNorm — it sits between blocks and head
            for param in vit.norm.parameters():
                param.requires_grad = True

        elif stage == 2:
            # Unfreeze next 25% — blocks 12-17
            unfreeze_from = int(num_blocks * 0.50)  # block 12
            unfreeze_to   = int(num_blocks * 0.75)  # block 18
            for i in range(unfreeze_from, unfreeze_to):
                for param in vit.blocks[i].parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        tqdm.write(f"🔓 Unfreeze stage {stage} applied | "
            f"Trainable: {trainable/1e6:.1f}M | Frozen: {frozen/1e6:.1f}M params")
    
    def forward(self, x):
        # from_pretrained backbone returns patch token sequence (B, N_patches, 1024)
        features = self.backbone(x, patch_size=self.patch_size)

        # Mean-pool patch tokens for spatially distributed wealth signal
        pooled = features.mean(dim=1)   # (B, 1024)
        return self.head(pooled)

