import importlib.util
from peft import LoraConfig, get_peft_model
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
    def __init__(self, image_size, bands=4, kind="reg"):
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
        flattened_dim = image_size // 4
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


class LateFusionHead(nn.Module):
    def __init__(self, embed_dim, meta_dim, out_features):
        super().__init__()
        self.meta_dim = meta_dim
        # 1. Compress the image features first
        self.image_compressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1)
        )
        # 2. Fuse with metadata, then map straight to output
        fuse_dim = 64 + meta_dim
        self.final_head = nn.Linear(fuse_dim, out_features)

    def forward(self, pooled, metadata=None):
        # Compress image features to a smaller bottleneck
        compressed_img = self.image_compressor(pooled)
        
        # Late fusion of covariates
        if metadata is not None and self.meta_dim > 0:
            fused = torch.cat([compressed_img, metadata], dim=1)
        else:
            fused = compressed_img
            
        return self.final_head(fused)



@register_model("scalemae")
class ScaleMAE(nn.Module):
    def __init__(self, image_size=224, bands=3, kind="reg", freeze_strategy="none", meta_dim=0):
        super().__init__()
        self.kind = kind
        self.meta_dim = meta_dim
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

        self.head = LateFusionHead(embed_dim, meta_dim, out_features)

        # Set LORA
        lora_config = LoraConfig(
            r=16,                     # The rank (8 or 16 is great)
            lora_alpha=16,           # Scaling factor
            target_modules=[
                "qkv",          # The fused attention matrix (no slicing required)
                "fc1",          # First linear layer of the MLP
                "fc2"           # Second linear layer of the MLP
            ],
            lora_dropout=0.05,
            bias="none",
            # Note: modules_to_save=["head"] is intentionally omitted.
            # self.head is a sibling of self.backbone (not a child), so PEFT cannot
            # find it when wrapping backbone. The head is saved/loaded explicitly via
            # torch.save(model.head.state_dict()) in train_model().
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x, metadata=None):
        # from_pretrained backbone returns patch token sequence (B, N_patches, 1024)
        features = self.backbone(x, patch_size=self.patch_size)

        # Mean-pool patch tokens for spatially distributed wealth signal
        pooled = features.mean(dim=1)   # (B, 1024)

        return self.head(pooled, metadata=metadata)


@register_model("scalemae_cov")
class ScaleMAECov(ScaleMAE):
    """ScaleMAE with tract-year ACS covariates fused into the head (diagnostic).

    Same backbone + LoRA + LateFusionHead as ``scalemae``; ``meta_dim`` is the
    number of covariates (``params["covariates"]``, see
    :mod:`src.data.covariates`). Warm-started from an imagery-only run with
    :func:`warm_start_from_checkpoint`, whose zero-initialized covariate
    weights make the model reproduce that run exactly at step 0.

    ``forward`` refuses metadata of the wrong width: every data path still
    carries the legacy (B, 1) ``dist_to_center`` column, and feeding it in
    place of the covariates would train silently on the wrong input.
    """

    def __init__(self, image_size=224, bands=3, kind="reg", freeze_strategy="none", meta_dim=2):
        if meta_dim < 1:
            raise ValueError("scalemae_cov needs meta_dim >= 1 (one per covariate)")
        super().__init__(image_size=image_size, bands=bands, kind=kind,
                         freeze_strategy=freeze_strategy, meta_dim=meta_dim)

    def forward(self, x, metadata=None):
        if metadata is None or metadata.dim() != 2 or metadata.shape[1] != self.meta_dim:
            got = None if metadata is None else tuple(metadata.shape)
            raise ValueError(f"scalemae_cov expects covariates of shape (B, {self.meta_dim}), "
                             f"got {got} — is params['covariates'] wired into this data path?")
        return super().forward(x, metadata=metadata)


# ==========================================
# 4. WARM START (imagery-only run -> covariate head)
# ==========================================

IMAGE_BOTTLENECK = 64   # LateFusionHead.image_compressor output width


def expand_head_state(base_state: dict, meta_dim: int) -> dict:
    """Head state of a ``meta_dim=0`` run, widened for ``meta_dim`` covariates.

    The covariate columns of ``final_head.weight`` are zero, so the widened head
    returns exactly the base head's output whatever the covariates are. A state
    that already has the target width is returned unchanged.
    """
    state = dict(base_state)
    w = state["final_head.weight"]
    if w.shape[1] == IMAGE_BOTTLENECK + meta_dim:
        return state
    if w.shape[1] != IMAGE_BOTTLENECK:
        raise ValueError(f"final_head.weight has {w.shape[1]} inputs; expected "
                         f"{IMAGE_BOTTLENECK} (imagery-only) or {IMAGE_BOTTLENECK + meta_dim}")
    state["final_head.weight"] = torch.cat(
        [w, torch.zeros(w.shape[0], meta_dim, dtype=w.dtype)], dim=1)
    return state


def _read_adapter_state(lora_dir):
    from pathlib import Path
    lora_dir = Path(lora_dir)
    st = lora_dir / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    return torch.load(lora_dir / "adapter_model.bin", map_location="cpu", weights_only=True)


def warm_start_from_checkpoint(model, head_path, lora_dir=None) -> dict:
    """Load an imagery-only run's best head (+ LoRA adapter) into ``model``.

    The adapter is loaded *into the existing PEFT parameters*
    (``set_peft_model_state_dict``) rather than by rebuilding the backbone with
    ``PeftModel.from_pretrained``: a rebuilt backbone is frozen by default and
    its parameters are not the ones an already-built optimizer holds. Call it
    before creating the optimizer anyway. Raises if any LoRA tensor fails to
    match, so a partially loaded adapter can never start a run.
    """
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    model.head.load_state_dict(expand_head_state(head, getattr(model, "meta_dim", 0)))
    report = {"head": str(head_path), "lora": None, "lora_tensors": 0}
    if lora_dir is not None:
        from peft import set_peft_model_state_dict
        adapter = _read_adapter_state(lora_dir)
        result = set_peft_model_state_dict(model.backbone, adapter)
        unexpected = [k for k in getattr(result, "unexpected_keys", []) if "lora_" in k]
        missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
        if unexpected or missing:
            raise RuntimeError(f"LoRA adapter {lora_dir} does not match the model: "
                               f"{len(missing)} missing, {len(unexpected)} unexpected "
                               f"(e.g. {(missing or unexpected)[:3]})")
        report["lora"] = str(lora_dir)
        report["lora_tensors"] = len(adapter)
    return report

