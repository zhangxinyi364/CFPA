import torch
import torch.nn as nn
import torch.nn.functional as F
from .otk.layers import OTKernel
from .models.transformer.encoders import EncoderLayer, EncoderLayer_BN
from .models.transformer.utils import PositionWiseFeedForward, PositionWiseFeedForward_BN
from .otk.utils import normalize
import math

class FeatureProcessor(nn.Module):
    def __init__(self, num_super_parts, feat_dim=1024):
        super().__init__()
        self.num_super_parts = num_super_parts
        self.feat_dim = feat_dim

        # OT Kernel
        self.otk_layer = OTKernel(
            in_dim=feat_dim,
            out_size=num_super_parts,
            heads=1,
            max_iter=100,
            log_domain=True
        )

        # Attention Layer Considering Geometric Distance Constraints
        self.geo_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            dropout=0.1
        )

        # Feature Fusion Module
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )



    def compute_distance_scores(self, features):
        """Calculate the distance score between features
        Args:
            features: [B, P, D] Batch Feature Vector
        Returns:
            scores: [B, P] Distance score for each feature
        """
        # Calculate the L2 distance between features
        x_square = torch.sum(features ** 2, dim=-1, keepdim=True)  # [B, P, 1]
        y_square = x_square.transpose(1, 2)  # [B, 1, P]
        
        dist_matrix = x_square + y_square - 2.0 * torch.bmm(features, features.transpose(1, 2))  # [B, P, P]
        dist_matrix = torch.clamp(dist_matrix, min=0.0)  
        
        # Calculate the total distance between each feature and all other features.
        dist_scores = torch.sum(torch.sqrt(dist_matrix + 1e-8), dim=2)  # [B, P]
        
        return dist_scores


    def forward(self, base_feat, contact_points=None, part_valid=None):
        B, P, D = base_feat.size()
        
        # 1. Calculate ranking scores based on feature distance
        dist_scores = self.compute_distance_scores(base_feat)  # [B, P]
        
        # 2. Sort by distance scores and construct the permutation matrix P
        _, indices = torch.sort(dist_scores, dim=1)  # [B, P]
        P_mat = torch.zeros(B, P, P, device=base_feat.device)
        batch_indices = torch.arange(B, device=base_feat.device).unsqueeze(1).expand(-1, P)
        P_mat[batch_indices, torch.arange(P).unsqueeze(0).expand(B, -1), indices] = 1
        
        # 3. Reordering Feature
        ordered_feat = torch.bmm(P_mat, base_feat)  # [B, P, D]

        self.ordered_feat = ordered_feat
        
        # 4. OT aggregation yields super-component features
        super_feat = self.otk_layer(ordered_feat, part_valid)  # [B, num_super_parts, D]
        
        # 4. Geometric Constraint Attention
        if contact_points is not None:
            # Modify the reordering method for contact_points
            ordered_contact_points = torch.zeros_like(contact_points)
            for b in range(B):
                ordered_contact_points[b] = contact_points[b, indices[b]][:, indices[b]]
            
            super_feat = super_feat.transpose(0, 1)
            
            # Constructing Attention Masks
            geo_mask = (ordered_contact_points[..., 0] > 0).float()
            geo_mask = geo_mask.mean(1) > 0.5
            geo_mask = geo_mask[:, :self.num_super_parts]
            
            # Apply attention
            attn_out = self.geo_attention(
                query=super_feat,
                key=super_feat,
                value=super_feat,
                key_padding_mask=geo_mask,
                need_weights=False
            )[0]
            
            super_feat = (super_feat + attn_out).transpose(0, 1)
        
        # 5. Feature Enhancement
        super_feat_expanded = self._expand_super_feat(super_feat, P)
        enhanced_feat = self.fusion_mlp(
            torch.cat([ordered_feat, super_feat_expanded], dim=-1)
        )
        
        return enhanced_feat, super_feat, P_mat, ordered_feat

    def _expand_super_feat(self, super_feat, num_parts):
        """Extend superpart feature to the number of original parts"""
        B, S, D = super_feat.size()

        # Add numerical stability processing
        scores = torch.matmul(super_feat, super_feat.transpose(-2, -1))  # [B,S,S]
        scores = scores / math.sqrt(self.feat_dim)  # resizing
        scores = F.softmax(scores, dim=-1)
        expanded = torch.matmul(scores, super_feat)  # [B,S,D]

        # Interpolate to the original part count
        expanded = F.interpolate(
            expanded.transpose(1, 2),  # [B,D,S]
            size=num_parts,
            mode='nearest',
            # align_corners=True  # Ensure consistency in interpolation
        ).transpose(1, 2)  # [B,P,D]

        return expanded
