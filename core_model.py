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
        
        # Episodic Graph State (Persists in VRAM)
        self.register_buffer('memory_nodes', torch.empty(0, feature_dim))  # V_cache
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long))
        
        # Running Class Prototypes for O(1) System 1
        self.register_buffer('class_sums', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Textual Anchors & Automatic Margins (Calculated offline)
        self.register_buffer('textual_anchors', torch.zeros(num_classes, feature_dim))
        self.register_buffer('tau_dist_margins', torch.zeros(num_classes)) # Class-conditional tau_dist
        
    def initialize_memory(self, support_patches, support_labels, text_features, tau_lambda=1.5):
        """
        Populates graph with few-shot support set & calculates automatic class-conditional margins.
        """
        self.textual_anchors = text_features
        self.memory_nodes = support_patches  # [N, D]
        self.memory_labels = support_labels  # [N]
        
        # Initialize System 1 Running Prototypes
        for c in range(self.num_classes):
            mask = (self.memory_labels == c)
            if mask.sum() > 0:
                self.class_sums[c] = self.memory_nodes[mask].sum(dim=0)
                self.class_counts[c] = mask.sum().float()
                
                # Calculate Automatic \tau_dist (Mean + Lambda * Std)
                class_patches = self.memory_nodes[mask]
                # Cosine distance to textual anchor
                dists = 1 - F.cosine_similarity(class_patches, self.textual_anchors[c].unsqueeze(0))
                mu_c, std_c = dists.mean(), dists.std()
                # If 1-shot, std is 0, add a small epsilon
                self.tau_dist_margins[c] = mu_c + (tau_lambda * std_c) if std_c > 0 else mu_c + 0.1
                
    def get_system1_prototypes(self):
        """Returns the mean-pooled visual memory nodes combined with textual anchors."""
        safe_counts = torch.clamp(self.class_counts, min=1.0).unsqueeze(1)
        visual_prototypes = self.class_sums / safe_counts
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        
        # Tip-Adapter style blending: Text + (Alpha * Mean Visual Memory)
        unified_prototypes = self.textual_anchors + (self.alpha * visual_prototypes)
        return F.normalize(unified_prototypes, dim=-1)

    def write_to_memory(self, patches, pred_class):
        """The 'Append' operation for Test-Time Adaptation."""
        self.memory_nodes = torch.cat([self.memory_nodes, patches], dim=0)
        self.memory_labels = torch.cat([self.memory_labels, torch.full((patches.size(0),), pred_class, device=patches.device)])
        
        # Update System 1 running metrics
        self.class_sums[pred_class] += patches.sum(dim=0)
        self.class_counts[pred_class] += patches.size(0)


# class System2Reasoner(nn.Module):
#     def __init__(self, feature_dim, top_k=50):
#         super().__init__()
#         self.top_k = top_k
        
#         # Modality-aware Graph Transformer (MGT) using PyG
#         # We define a simple heterogeneous setup: 'memory' nodes to 'test' patches
#         self.hgt_conv = HGTConv(
#             in_channels={'test': feature_dim, 'memory': feature_dim},
#             out_channels=feature_dim,
#             metadata=(['test', 'memory'], [('memory', 'interacts', 'test')]),
#             heads=4,
#             # group='sum'
#         )
        
#         # Inductive Token Evidence Scorer (Projects patch to an evidence scalar)
#         self.evidence_scorer = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.ReLU(),
#             nn.Linear(feature_dim // 2, 1)
#         )

#     def compute_exact_topk_subgraph(self, test_patches, memory_nodes_gpu):
#         """Exact dense-to-sparse routing WITH Dynamic Subgraph Extraction to save VRAM."""
#         P = test_patches.size(0)
        
#         # Exact dot product similarity against FULL memory
#         sim_matrix = torch.matmul(test_patches, memory_nodes_gpu.T) # [P, N]
        
#         K = min(self.top_k, memory_nodes_gpu.size(0))
#         _, topk_indices = torch.topk(sim_matrix, k=K, dim=1) # [P, K]
        
#         # --- THE FIX: DYNAMIC SUBGRAPH EXTRACTION ---
#         flat_topk = topk_indices.reshape(-1)
#         # Find the unique memory nodes actually needed for this pass
#         unique_mem_idx, inverse_idx = torch.unique(flat_topk, return_inverse=True)
        
#         # Extract only the active nodes (Shrinks memory size from ~300k to max 9800)
#         active_memory_nodes = memory_nodes_gpu[unique_mem_idx]
        
#         # Build PyG Edge Index [2, P * K]
#         test_node_idx = torch.arange(P, device=test_patches.device).view(-1, 1).expand(-1, K).reshape(-1)
        
#         # The new edges map from the small active memory pool to the test patches
#         dynamic_edge_index = torch.stack([inverse_idx, test_node_idx], dim=0)
        
#         return dynamic_edge_index, active_memory_nodes

#     def token_evidence_pooling(self, patches):
#         """Assigns high weights to discriminative patches, suppresses noisy background."""
#         evidence_logits = self.evidence_scorer(patches) # [P, 1]
#         evidence_weights = F.softmax(evidence_logits, dim=0)
#         global_feature = (patches * evidence_weights).sum(dim=0, keepdim=True)
#         return F.normalize(global_feature, dim=-1)



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
        """
        The Causally Strict Inference Pipeline.
        """
        # --- Step 3.1: System 1 (Fast Pass) ---
        sys1_prototypes = self.memory.get_system1_prototypes()
        sys1_logits = 100.0 * test_global @ sys1_prototypes.T # CLIP scale factor
        entropy = self.calculate_entropy(sys1_logits)
        
        # Initial prediction
        pred_class = sys1_logits.argmax(dim=-1).item()
        
        # --- Step 3.2 & 3.3: Router & Two-Factor Gating (Fast Exit) ---
        if entropy <= self.tau_conf:
            # Check Anchor Gate (Cosine Distance constraint)
            dist_to_anchor = 1 - F.cosine_similarity(test_global, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            
            if dist_to_anchor <= margin:
                # Passes BOTH gates -> Safely update memory
                self.memory.write_to_memory(test_patches, pred_class)
            
            return sys1_logits, 0 # Return logits and steps taken (0)

        current_patches = test_patches
        step = 0
        final_logits = sys1_logits
        
        # CPU Offloading fetch
        memory_nodes_gpu = self.memory.memory_nodes.to(test_patches.device, non_blocking=True)
        
        while entropy > self.tau_conf and step < max_steps:
            
            # --- THE FIX: One-Line Parameter-Free Graph Reasoning ---
            global_updated, current_patches = self.system2(current_patches, memory_nodes_gpu)
            
            # 3. Router Re-evaluation
            final_logits = 100.0 * global_updated @ sys1_prototypes.T
            entropy = self.calculate_entropy(final_logits)
            step += 1
            pred_class = final_logits.argmax(dim=-1).item()

        # Try to write to memory one last time if System 2 solved it
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(global_updated, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            if dist_to_anchor <= margin:
                self.memory.write_to_memory(current_patches, pred_class)

        return final_logits, step