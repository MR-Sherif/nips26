import os
import torch
import clip
import math
from tqdm import tqdm

from utils import clip_classifier
from datasets import build_dataset
from datasets.utils import build_data_loader
from core_model import ContinuousEpisodicVLM
import torch.nn.functional as F

# --- Setup & Hyperparameters ---
DATASET_NAME = 'fgvc' # Swap for your 11 datasets
DATA_PATH = '/home/user/codex/FewShot/TipAdapter/data'
SHOTS = 16
BACKBONE = 'RN50' # or ViT-B/16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# =============================================================================
# 1. UNIVERSAL PATCH FOR SPATIAL FEATURES (Hybrid Strategy)
# =============================================================================

class SpatialFeatureHook:
    """ Context manager to capture outputs from ViT layers """
    def __init__(self, module):
        self.module = module
        self.hook_handle = None
        self.captured_output = None

    def hook_fn(self, module, input, output):
        self.captured_output = output

    def enable(self):
        self.hook_handle = self.module.register_forward_hook(self.hook_fn)
    
    def disable(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def get_data(self):
        return self.captured_output


def patch_open_clip_model(model):
    """
    Applies the appropriate patch based on architecture using Forward Hooks.
    Updated to upsample ResNet spatial features to 14x14.
    """
    original_visual = model.visual
    original_encode_image = model.encode_image

    # ------------------------------------------------------------------
    # CASE A: Vision Transformer (ViT) - (Unchanged)
    # ------------------------------------------------------------------
    if hasattr(original_visual, 'transformer'):
        target_layer = original_visual.ln_post
        hook = SpatialFeatureHook(target_layer)
        hook.enable() 

        def new_encode_image_vit(self, image):
            global_feat = original_encode_image(image)
            full_sequence = hook.get_data() # (N, L, D)
            
            if hasattr(original_visual, 'proj') and original_visual.proj is not None:
                full_sequence = full_sequence @ original_visual.proj
            
            spatial_feat = full_sequence[:, 1:, :] 
            return global_feat, spatial_feat

        import types
        model.encode_image = types.MethodType(new_encode_image_vit, model)

    # ------------------------------------------------------------------
    # CASE B: ResNet (RN50) -> Updated to Upsample to 14x14
    # ------------------------------------------------------------------
    elif hasattr(original_visual, 'attnpool'):
        # Hook the last ResNet layer
        target_layer = original_visual.layer4
        hook = SpatialFeatureHook(target_layer)
        hook.enable()

        def new_encode_image_resnet(self, image):
            # 1. Run standard forward to trigger hook and get global features
            global_feat = original_encode_image(image)

            # 2. Retrieve spatial feature map: (N, 2048, 7, 7)
            x_raw = hook.get_data() 
            
            # 3. Apply Attention Pooling logic to transform features to embedding space
            attnpool = original_visual.attnpool

            # Flatten: (N, C, H, W) -> (HW, N, C) -> (49, N, 2048)
            x = x_raw.reshape(x_raw.shape[0], x_raw.shape[1], -1).permute(2, 0, 1)
            
            # Add mean token
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

            # Add Positional Embedding (standard CLIP RN50 pos_embed is for 7x7)
            pos_embed = attnpool.positional_embedding[:, None, :].to(x.dtype)
            if x.shape[0] != pos_embed.shape[0]:
                pos_embed = pos_embed[:x.shape[0]]
            x = x + pos_embed

            # Run Multi-Head Attention
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=attnpool.num_heads,
                q_proj_weight=attnpool.q_proj.weight,
                k_proj_weight=attnpool.k_proj.weight,
                v_proj_weight=attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]),
                bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
                out_proj_weight=attnpool.c_proj.weight,
                out_proj_bias=attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=attnpool.training,
                need_weights=False
            )

            # -------------------------------------------------------
            # UP-SAMPLING LOGIC
            # -------------------------------------------------------
            # x is currently (50, N, D) -> [Global, Spatial...]
            spatial_tokens = x[1:] # (49, N, D)
            
            # Permute to (N, D, 49)
            spatial_tokens = spatial_tokens.permute(1, 2, 0)
            
            # Reshape to grid (N, D, 7, 7)
            side = int(math.sqrt(spatial_tokens.shape[-1]))
            feature_map = spatial_tokens.view(spatial_tokens.shape[0], spatial_tokens.shape[1], side, side)
            
            # INTERPOLATE: 7x7 -> 14x14
            # This simulates creating patches at a higher resolution
            feature_map = F.interpolate(feature_map, size=(14, 14), mode='bicubic', align_corners=False)
            
            # Flatten back: (N, D, 14, 14) -> (N, D, 196) -> (N, 196, D)
            spatial_feat = feature_map.flatten(2).permute(0, 2, 1)

            return global_feat, spatial_feat

        import types
        model.encode_image = types.MethodType(new_encode_image_resnet, model)

    else:
        raise ValueError("Unknown OpenCLIP architecture.")

    return model


# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

def pre_load_features(cfg, split, model, loader):
    global_features, spatial_features, labels = [], [], []
    print(f"Extracting Global + Spatial features for {split}...")
    
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            
            global_feat, spatial_feat = model.encode_image(images)
            
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            spatial_feat /= spatial_feat.norm(dim=-1, keepdim=True)

            global_features.append(global_feat)
            spatial_features.append(spatial_feat)
            labels.append(target)

    global_features = torch.cat(global_features)
    spatial_features = torch.cat(spatial_features)
    labels = torch.cat(labels)
    
    return global_features, spatial_features, labels


def extract_clip_features(dataloader, model, device):
    global_features, patch_features, labels = [], [], []
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            target = target.to(device)
            
            image_features, dense_patches = model.encode_image(images) 
            
            # --- CRITICAL FIX: L2 NORMALIZE NATIVELY ---
            image_features = F.normalize(image_features, dim=-1)
            dense_patches = F.normalize(dense_patches, dim=-1)
            
            global_features.append(image_features)
            patch_features.append(dense_patches)
            labels.append(target)
            
    return torch.cat(global_features), torch.cat(patch_features), torch.cat(labels)

def main():
    print(f"Loading CLIP {BACKBONE}...")
    clip_model, preprocess = clip.load(BACKBONE, device=DEVICE)
    clip_model.eval()
    clip_model = patch_open_clip_model(clip_model)
    
    # 1. Load Dataset using Tip-Adapter's boilerplate
    print(f"Preparing {DATASET_NAME} dataset...")
    dataset = build_dataset(DATASET_NAME, DATA_PATH, SHOTS)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, is_train=False, tfm=preprocess)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess) # Streaming size 1
    
    print("Extracting Textual Anchors...")
    text_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    text_features = text_features.t().float()
    
    # 2. Extract Few-Shot Support Set
    print("Extracting Few-Shot Support Set (Initialization)...")
    support_global, support_patches, support_labels = extract_clip_features(train_loader, clip_model, DEVICE)
    
    # Cast to float
    support_global = support_global.float()
    support_labels = support_labels # Shape [N]
    
    # Flatten support patches to shape [N*P, D] for the memory bank
    B, P, D = support_patches.shape
    support_patches_flat = support_patches.view(-1, D).float()
    support_labels_flat = support_labels.unsqueeze(1).expand(-1, P).reshape(-1)
    
    # 3. Initialize Model
    ALPHA = 1.5 
    print("Initializing Continuous Episodic Graph Memory...")
    model = ContinuousEpisodicVLM(feature_dim=D, num_classes=len(dataset.classnames), tau_conf=0.8)
    model.memory.alpha = ALPHA
    model.to(DEVICE)
    model.eval() 
    
    # Pass ALL the arguments required for the fixed Global/Local separation
    # (NEW) Mean-Pooling Initialization
    # (NEW) Mean-Pooling Initialization
    model.memory.initialize_memory(
        support_patches=support_patches_flat, 
        support_labels=support_labels_flat, 
        text_features=text_features
    )
    
    # 4. The Streaming Inference Loop
    print("Starting Continuous Test-Time Adaptation Stream...")
    correct = 0
    total = 0
    sys2_calls = 0
    
    with torch.no_grad(): # Strict compliance: NO optimizer
        for images, target in tqdm(test_loader, desc="Streaming Inference"):
            images = images.to(DEVICE)
            target = target.to(DEVICE)
            
            # Extract test features
            test_global, test_patches = clip_model.encode_image(images)
            test_global = F.normalize(test_global.squeeze(0).float(), dim=-1) 
            test_patches = F.normalize(test_patches.squeeze(0).float(), dim=-1)
            
            # Forward Pass through Dynamic Topology
            logits, steps = model(test_global, test_patches, max_steps=3)
            
            if steps > 0:
                sys2_calls += 1
                
            pred = logits.argmax(dim=-1)
            if pred == target.item():
                correct += 1
            total += 1
            
    acc = (correct / total) * 100
    print(f"\nFinal TTA Accuracy: {acc:.2f}%")
    print(f"System 2 Activated on {sys2_calls}/{total} difficult samples.")
    print(f"Final Graph Memory Size: {model.memory.memory_nodes.size(0)} nodes")

if __name__ == '__main__':
    main()