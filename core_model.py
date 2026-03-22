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


class System2Reasoner(nn.Module):
    def __init__(self, feature_dim, top_k=50):
        super().__init__()
        self.top_k = top_k
        
        # Modality-aware Graph Transformer (MGT) using PyG
        # We define a simple heterogeneous setup: 'memory' nodes to 'test' patches
        self.hgt_conv = HGTConv(
            in_channels={'test': feature_dim, 'memory': feature_dim},
            out_channels=feature_dim,
            metadata=(['test', 'memory'], [('memory', 'interacts', 'test')]),
            heads=4,
            group='sum'
        )
        
        # Inductive Token Evidence Scorer (Projects patch to an evidence scalar)
        self.evidence_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def compute_exact_topk_edges(self, test_patches, memory_nodes):
        """Exact dense-to-sparse routing using pure tensor cores (Bypasses ANN accuracy drop)."""
        P = test_patches.size(0)
        
        # Exact dot product similarity
        sim_matrix = torch.matmul(test_patches, memory_nodes.T) # [P, N]
        
        # Avoid K > N during the very first few test samples
        K = min(self.top_k, memory_nodes.size(0))
        _, topk_indices = torch.topk(sim_matrix, k=K, dim=1) # [P, K]
        
        # Build PyG Edge Index [2, P * K]
        test_node_idx = torch.arange(P, device=test_patches.device).view(-1, 1).expand(-1, K).reshape(-1)
        memory_node_idx = topk_indices.reshape(-1)
        
        return torch.stack([memory_node_idx, test_node_idx], dim=0)

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

        # --- Step 3.4: System 2 (Iterative Graph Reasoner) ---
        current_patches = test_patches
        step = 0
        final_logits = sys1_logits
        
        while entropy > self.tau_conf and step < max_steps:
            # 1. Dynamic Edge Formation (Read Query)
            edge_index = self.system2.compute_exact_topk_edges(current_patches, self.memory.memory_nodes)
            
            # 2. HGT Message Passing
            x_dict = {'test': current_patches, 'memory': self.memory.memory_nodes}
            edge_index_dict = {('memory', 'interacts', 'test'): edge_index}
            
            updated_dict = self.system2.hgt_conv(x_dict, edge_index_dict)
            current_patches = current_patches + updated_dict['test'] # Residual connection
            current_patches = F.normalize(current_patches, dim=-1)
            
            # 3. Step 3.5: Final Resolution via Token Evidence
            global_updated = self.system2.token_evidence_pooling(current_patches)
            final_logits = 100.0 * global_updated @ sys1_prototypes.T
            
            # 4. Router Re-evaluation
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