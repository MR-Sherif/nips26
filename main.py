import os
import torch
import clip
from tqdm import tqdm

from datasets import build_dataset
from datasets.utils import build_data_loader
from core_model import ContinuousEpisodicVLM

# --- Setup & Hyperparameters ---
DATASET_NAME = 'imagenet' # Swap for your 11 datasets
DATA_PATH = './data'
SHOTS = 16
BACKBONE = 'RN50' # or ViT-B/16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_clip_features(dataloader, model, device):
    """Extracts both global [CLS] token and fine-grained patch tokens."""
    global_features, patch_features, labels = [], [], []
    
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            target = target.to(device)
            
            # Note: Depending on your specific CLIP model (ViT vs ResNet), 
            # you might need to modify model.encode_image to return dense patches.
            # Assuming ViT here where sequence output is [B, P, D] and global is [B, D]
            image_features, dense_patches = model.encode_image_dense(images) 
            
            global_features.append(image_features)
            patch_features.append(dense_patches)
            labels.append(target)
            
    return torch.cat(global_features), torch.cat(patch_features), torch.cat(labels)

def main():
    print(f"Loading CLIP {BACKBONE}...")
    clip_model, preprocess = clip.load(BACKBONE, device=DEVICE)
    clip_model.eval()
    
    # 1. Load Dataset using Tip-Adapter's boilerplate
    print(f"Preparing {DATASET_NAME} dataset...")
    dataset = build_dataset(DATASET_NAME, DATA_PATH, SHOTS)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, is_train=False, tfm=preprocess)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess) # Streaming size 1
    
    # Extract Textual Anchors (using tip-adapter prompt templates)
    # (Assuming you have a function `get_text_features` similar to Tip-Adapter)
    # text_features = get_text_features(clip_model, dataset.classnames, dataset.template)
    text_features = torch.randn(len(dataset.classnames), 1024).to(DEVICE) # Placeholder
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    
    # 2. Extract Few-Shot Support Set
    print("Extracting Few-Shot Support Set (Initialization)...")
    _, support_patches, support_labels = extract_clip_features(train_loader, clip_model, DEVICE)
    
    # Flatten support patches to shape [N, D] for the memory bank
    # B = batch, P = patches, D = dim
    B, P, D = support_patches.shape
    support_patches_flat = support_patches.view(-1, D)
    support_labels_flat = support_labels.unsqueeze(1).expand(-1, P).reshape(-1)
    
    # 3. Initialize Model
    print("Initializing Continuous Episodic Graph Memory...")
    model = ContinuousEpisodicVLM(feature_dim=D, num_classes=len(dataset.classnames), tau_conf=0.8)
    model.to(DEVICE)
    model.eval() # MUST REMAIN IN EVAL. No Backprop!
    
    model.memory.initialize_memory(support_patches_flat, support_labels_flat, text_features)
    
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
            test_global, test_patches = clip_model.encode_image_dense(images)
            test_global = test_global.squeeze(0) # [D]
            test_patches = test_patches.squeeze(0) # [P, D]
            
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