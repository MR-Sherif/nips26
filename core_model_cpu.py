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
        
        # --- CPU OFFLOADING (Host RAM) ---
        # We do NOT use register_buffer for these, so model.to('cuda') ignores them.
        self.memory_nodes = torch.empty(0, feature_dim, device='cpu')
        self.memory_labels = torch.empty(0, dtype=torch.long, device='cpu')
        
        # --- GPU CACHE (System 1) ---
        # Running Class Prototypes stay on GPU for instant O(1) System 1
        self.register_buffer('class_sums', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Textual Anchors & Margins stay on GPU
        self.register_buffer('textual_anchors', torch.zeros(num_classes, feature_dim))
        self.register_buffer('tau_dist_margins', torch.zeros(num_classes)) 
        
    def initialize_memory(self, support_patches, support_labels, text_features, tau_lambda=1.5):
        """Populates graph with few-shot support set."""
        self.textual_anchors = text_features
        
        # Force the massive graph to stay on Host RAM
        self.memory_nodes = support_patches.cpu()  
        self.memory_labels = support_labels.cpu()  
        
        # Initialize System 1 Running Prototypes
        for c in range(self.num_classes):
            mask_cpu = (self.memory_labels == c)
            if mask_cpu.sum() > 0:
                # Temporarily fetch to GPU to compute stats
                class_patches_gpu = self.memory_nodes[mask_cpu].to(self.textual_anchors.device)
                
                self.class_sums[c] = class_patches_gpu.sum(dim=0)
                self.class_counts[c] = mask_cpu.sum().float()
                
                # Calculate Automatic \tau_dist
                dists = 1 - F.cosine_similarity(class_patches_gpu, self.textual_anchors[c].unsqueeze(0))
                mu_c, std_c = dists.mean(), dists.std()
                self.tau_dist_margins[c] = mu_c + (tau_lambda * std_c) if std_c > 0 else mu_c + 0.1
                
    def get_system1_prototypes(self):
        """Returns the mean-pooled visual memory nodes combined with textual anchors."""
        safe_counts = torch.clamp(self.class_counts, min=1.0).unsqueeze(1)
        visual_prototypes = self.class_sums / safe_counts
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        
        unified_prototypes = self.textual_anchors + (self.alpha * visual_prototypes)
        return F.normalize(unified_prototypes, dim=-1)

    def write_to_memory(self, patches, pred_class):
        """The 'Append' operation: Updates GPU metrics, saves tensors to CPU RAM."""
        # 1. Update System 1 running metrics on the GPU
        self.class_sums[pred_class] += patches.sum(dim=0)
        self.class_counts[pred_class] += patches.size(0)
        
        # 2. Append full tokens to Host RAM
        patches_cpu = patches.cpu()
        labels_cpu = torch.full((patches.size(0),), pred_class, dtype=torch.long, device='cpu')
        
        self.memory_nodes = torch.cat([self.memory_nodes, patches_cpu], dim=0)
        self.memory_labels = torch.cat([self.memory_labels, labels_cpu], dim=0)


class System2Reasoner(nn.Module):
    def __init__(self, feature_dim, top_k=50):
        super().__init__()
        self.top_k = top_k
        
        self.hgt_conv = HGTConv(
            in_channels={'test': feature_dim, 'memory': feature_dim},
            out_channels=feature_dim,
            metadata=(['test', 'memory'], [('memory', 'interacts', 'test')]),
            heads=4,
            # group='sum'
        )
        
        self.evidence_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def compute_exact_topk_subgraph(self, test_patches, memory_nodes_gpu):
        """Exact dense-to-sparse routing WITH Dynamic Subgraph Extraction to save VRAM."""
        P = test_patches.size(0)
        
        # Exact dot product similarity against FULL memory
        sim_matrix = torch.matmul(test_patches, memory_nodes_gpu.T) # [P, N]
        
        K = min(self.top_k, memory_nodes_gpu.size(0))
        _, topk_indices = torch.topk(sim_matrix, k=K, dim=1) # [P, K]
        
        # --- THE FIX: DYNAMIC SUBGRAPH EXTRACTION ---
        flat_topk = topk_indices.reshape(-1)
        # Find the unique memory nodes actually needed for this pass
        unique_mem_idx, inverse_idx = torch.unique(flat_topk, return_inverse=True)
        
        # Extract only the active nodes (Shrinks memory size from ~300k to max 9800)
        active_memory_nodes = memory_nodes_gpu[unique_mem_idx]
        
        # Build PyG Edge Index [2, P * K]
        test_node_idx = torch.arange(P, device=test_patches.device).view(-1, 1).expand(-1, K).reshape(-1)
        
        # The new edges map from the small active memory pool to the test patches
        dynamic_edge_index = torch.stack([inverse_idx, test_node_idx], dim=0)
        
        return dynamic_edge_index, active_memory_nodes

    def token_evidence_pooling(self, patches):
        """Assigns high weights to discriminative patches, suppresses noisy background."""
        evidence_logits = self.evidence_scorer(patches) # [P, 1]
        evidence_weights = F.softmax(evidence_logits, dim=0)
        global_feature = (patches * evidence_weights).sum(dim=0, keepdim=True)
        return F.normalize(global_feature, dim=-1)


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
        # --- Step 3.1: System 1 (Fast Pass, 100% on GPU) ---
        sys1_prototypes = self.memory.get_system1_prototypes()
        sys1_logits = 100.0 * test_global @ sys1_prototypes.T 
        entropy = self.calculate_entropy(sys1_logits)
        
        pred_class = sys1_logits.argmax(dim=-1).item()
        
        # --- Step 3.2 & 3.3: Router & Two-Factor Gating (Fast Exit) ---
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(test_global, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            
            if dist_to_anchor <= margin:
                self.memory.write_to_memory(test_patches, pred_class)
            
            return sys1_logits, 0 

        # --- Step 3.4: System 2 (Iterative Graph Reasoner) ---
        # --- Step 3.4: System 2 (Iterative Graph Reasoner) ---
        current_patches = test_patches
        step = 0
        final_logits = sys1_logits
        
        # Fetch the entire memory graph from Host RAM to VRAM just ONCE for this loop
        memory_nodes_gpu = self.memory.memory_nodes.to(test_patches.device, non_blocking=True)
        
        while entropy > self.tau_conf and step < max_steps:
            # 1. Dynamic Edge Formation & Subgraph Extraction (THE FIX)
            edge_index, active_memory = self.system2.compute_exact_topk_subgraph(current_patches, memory_nodes_gpu)
            
            # 2. HGT Message Passing (Now running on a tiny fraction of nodes)
            x_dict = {'test': current_patches, 'memory': active_memory}
            edge_index_dict = {('memory', 'interacts', 'test'): edge_index}
            
            updated_dict = self.system2.hgt_conv(x_dict, edge_index_dict)
            current_patches = current_patches + updated_dict['test'] # Residual connection
            current_patches = F.normalize(current_patches, dim=-1)
            
            # 3. Step 3.5: Final Resolution
            global_updated = self.system2.token_evidence_pooling(current_patches)
            final_logits = 100.0 * global_updated @ sys1_prototypes.T
            
            # 4. Router Re-evaluation
            entropy = self.calculate_entropy(final_logits)
            step += 1
            pred_class = final_logits.argmax(dim=-1).item()

        # Write check
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(global_updated, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            if dist_to_anchor <= margin:
                self.memory.write_to_memory(current_patches, pred_class)

        return final_logits, step