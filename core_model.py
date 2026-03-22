import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class TopologicalGraphMemory(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # --- GPU VRAM STATE ---
        self.register_buffer('memory_nodes', torch.empty(0, feature_dim))  
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long))
        
        # Running Class Prototypes for O(1) System 1 (Must be GLOBAL tokens)
        self.register_buffer('class_sums', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Textual Anchors & Margins
        self.register_buffer('textual_anchors', torch.zeros(num_classes, feature_dim))
        self.register_buffer('tau_dist_margins', torch.zeros(num_classes)) 
        
    def initialize_memory(self, support_global, support_labels, support_patches, support_patches_labels, text_features, tau_lambda=1.5):
        self.textual_anchors = text_features
        self.memory_nodes = support_patches  # [N*P, D]
        self.memory_labels = support_patches_labels  # [N*P]
        
        # FIX: Initialize System 1 Prototypes using ONLY GLOBAL features!
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_global = support_global[mask]
                self.class_sums[c] = class_global.sum(dim=0)
                self.class_counts[c] = mask.sum().float()
                
                # Automatic margin based on global features
                dists = 1 - F.cosine_similarity(class_global, self.textual_anchors[c].unsqueeze(0))
                mu_c, std_c = dists.mean(), dists.std()
                self.tau_dist_margins[c] = mu_c + (tau_lambda * std_c) if std_c > 0 else mu_c + 0.1
                
    def get_system1_prototypes(self):
        safe_counts = torch.clamp(self.class_counts, min=1.0).unsqueeze(1)
        visual_prototypes = self.class_sums / safe_counts
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        
        unified_prototypes = self.textual_anchors + (self.alpha * visual_prototypes)
        return F.normalize(unified_prototypes, dim=-1)

    def write_to_memory(self, test_global, test_patches, pred_class):
        """Appends RAW patches to Graph, and RAW global to Prototypes."""
        self.memory_nodes = torch.cat([self.memory_nodes, test_patches], dim=0)
        self.memory_labels = torch.cat([self.memory_labels, torch.full((test_patches.size(0),), pred_class, device=test_patches.device)])
        
        # FIX: Update System 1 using ONLY the global token
        self.class_sums[pred_class] += test_global.squeeze()
        self.class_counts[pred_class] += 1


class System2Reasoner(nn.Module):
    def __init__(self, feature_dim, top_k=50, tau=0.02):
        super().__init__()
        self.top_k = top_k
        self.tau = tau
        # ALL RANDOM WEIGHTS REMOVED. 100% Training-Free.

    def forward(self, test_patches, memory_nodes_gpu):
        P = test_patches.size(0)
        
        # 1. Exact Dot Product Similarity
        sim_matrix = torch.matmul(test_patches, memory_nodes_gpu.T) # [P, N]
        
        # 2. Extract Top-K Nodes per Patch
        K = min(self.top_k, memory_nodes_gpu.size(0))
        topk_sim, topk_indices = torch.topk(sim_matrix, k=K, dim=1) # [P, K]
        
        # 3. Softmax over Top-K to get Attention Weights
        attn_weights = F.softmax(topk_sim / self.tau, dim=1) # [P, K]
        
        # 4. Gather the actual memory nodes
        gathered_memory = memory_nodes_gpu[topk_indices] # [P, K, D]
        
        # 5. Parameter-Free Message Passing (Weighted Sum)
        messages = (attn_weights.unsqueeze(-1) * gathered_memory).sum(dim=1) # [P, D]
        
        # Residual Connection
        updated_patches = test_patches + messages
        updated_patches = F.normalize(updated_patches, dim=-1)
        
        # 6. Training-Free Token Evidence 
        # (Use the highest similarity score to the memory graph as evidence weight)
        evidence_scores = topk_sim[:, 0] 
        evidence_weights = F.softmax(evidence_scores / self.tau, dim=0).unsqueeze(-1) # [P, 1]
        
        global_feature = (updated_patches * evidence_weights).sum(dim=0, keepdim=True)
        return F.normalize(global_feature, dim=-1), updated_patches


class ContinuousEpisodicVLM(nn.Module):
    def __init__(self, feature_dim, num_classes, tau_conf=0.8):
        super().__init__()
        self.memory = TopologicalGraphMemory(feature_dim, num_classes)
        self.system2 = System2Reasoner(feature_dim)
        self.tau_conf = tau_conf

    def calculate_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()

    @torch.no_grad()
    def forward(self, test_global, test_patches, max_steps=3):
        # --- Step 3.1: System 1 (Fast Pass) ---
        sys1_prototypes = self.memory.get_system1_prototypes()
        sys1_logits = 100.0 * test_global @ sys1_prototypes.T 
        entropy = self.calculate_entropy(sys1_logits)
        
        pred_class = sys1_logits.argmax(dim=-1).item()
        
        # --- Fast Exit ---
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(test_global, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            if dist_to_anchor <= margin:
                # FIX: Write original global and patches
                self.memory.write_to_memory(test_global, test_patches, pred_class)
            return sys1_logits, 0 

        # --- Step 3.4: System 2 ---
        current_patches = test_patches
        step = 0
        final_logits = sys1_logits
        memory_nodes_gpu = self.memory.memory_nodes # (Assuming standard VRAM implementation here)
        
        while entropy > self.tau_conf and step < max_steps:
            global_updated, current_patches = self.system2(current_patches, memory_nodes_gpu)
            final_logits = 100.0 * global_updated @ sys1_prototypes.T
            
            entropy = self.calculate_entropy(final_logits)
            step += 1
            pred_class = final_logits.argmax(dim=-1).item()

        # Try to write to memory one last time
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(global_updated, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            if dist_to_anchor <= margin:
                # FIX: Append RAW features to avoid semantic drift!
                self.memory.write_to_memory(test_global, test_patches, pred_class)

        return final_logits, step