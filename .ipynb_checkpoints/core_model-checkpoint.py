import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalGraphMemory(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # --- PURE VRAM MODE (GPU) ---
        # Using register_buffer so model.to('cuda') handles these automatically
        self.register_buffer('memory_nodes', torch.empty(0, feature_dim))
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long))
        
        # --- GPU CACHE (System 1) ---
        # YOUR ROBUST MEAN-POOLING PROTOTYPES
        self.register_buffer('class_sums', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        self.register_buffer('textual_anchors', torch.zeros(num_classes, feature_dim))
        self.register_buffer('tau_dist_margins', torch.zeros(num_classes)) 
        
    def initialize_memory(self, support_patches, support_labels, text_features, tau_lambda=1.5):
        """Populates graph with few-shot support set directly in VRAM."""
        self.textual_anchors = text_features
        
        # Keep everything on the GPU
        self.memory_nodes = support_patches  
        self.memory_labels = support_labels  
        
        for c in range(self.num_classes):
            mask = (self.memory_labels == c)
            if mask.sum() > 0:
                class_patches_gpu = self.memory_nodes[mask]
                self.class_sums[c] = class_patches_gpu.sum(dim=0)
                self.class_counts[c] = mask.sum().float()
                
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
        """Your robust mean-pooling TTA update, kept strictly in VRAM!"""
        self.class_sums[pred_class] += patches.sum(dim=0)
        self.class_counts[pred_class] += patches.size(0)
        
        # Create labels directly on the GPU
        labels_gpu = torch.full((patches.size(0),), pred_class, dtype=torch.long, device=patches.device)
        
        self.memory_nodes = torch.cat([self.memory_nodes, patches], dim=0)
        self.memory_labels = torch.cat([self.memory_labels, labels_gpu], dim=0)


class System2Reasoner(nn.Module):
    def __init__(self, feature_dim, top_k=50, tau=0.02):
        super().__init__()
        self.top_k = top_k
        self.tau = tau
        # 100% Training-Free Now!

    def message_passing(self, test_patches, memory_nodes_gpu):
        """Parameter-Free equivalent of HGTConv."""
        P = test_patches.size(0)
        sim_matrix = torch.matmul(test_patches, memory_nodes_gpu.T) # [P, N]
        
        K = min(self.top_k, memory_nodes_gpu.size(0))
        topk_sim, topk_indices = torch.topk(sim_matrix, k=K, dim=1) # [P, K]
        
        attn_weights = F.softmax(topk_sim / self.tau, dim=1) 
        gathered_memory = memory_nodes_gpu[topk_indices] # [P, K, D]
        messages = (attn_weights.unsqueeze(-1) * gathered_memory).sum(dim=1) # [P, D]
        
        return messages

    def token_evidence_pooling(self, patches, text_anchors):
        """Parameter-Free equivalent of evidence_scorer (uses semantics, not random weights)"""
        # Compare patches directly to text anchors to find foreground objects
        evidence_logits = (patches @ text_anchors.T).max(dim=-1).values.unsqueeze(1) # [P, 1]
        evidence_weights = F.softmax(evidence_logits / self.tau, dim=0)
        
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
        sys1_prototypes = self.memory.get_system1_prototypes()
        sys1_logits = 100.0 * test_global @ sys1_prototypes.T 
        entropy = self.calculate_entropy(sys1_logits)
        
        pred_class = sys1_logits.argmax(dim=-1).item()
        
        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(test_global, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            
            if dist_to_anchor <= margin:
                self.memory.write_to_memory(test_patches, pred_class)
            
            return sys1_logits, 0 

        current_patches = test_patches
        step = 0
        final_logits = sys1_logits
        
        # PURE VRAM: No CPU-to-GPU transfer needed anymore!
        memory_nodes_gpu = self.memory.memory_nodes
        
        while entropy > self.tau_conf and step < max_steps:
            # 1 & 2. Parameter-Free Message Passing
            messages = self.system2.message_passing(current_patches, memory_nodes_gpu)
            current_patches = current_patches + messages
            current_patches = F.normalize(current_patches, dim=-1)
            
            # 3. Step 3.5: Final Resolution via Semantic Token Evidence
            global_updated = self.system2.token_evidence_pooling(current_patches, self.memory.textual_anchors)
            final_logits = 100.0 * global_updated @ sys1_prototypes.T
            
            entropy = self.calculate_entropy(final_logits)
            step += 1
            pred_class = final_logits.argmax(dim=-1).item()

        if entropy <= self.tau_conf:
            dist_to_anchor = 1 - F.cosine_similarity(global_updated, self.memory.textual_anchors[pred_class].unsqueeze(0)).item()
            margin = self.memory.tau_dist_margins[pred_class].item()
            if dist_to_anchor <= margin:
                self.memory.write_to_memory(current_patches, pred_class)

        return final_logits, step