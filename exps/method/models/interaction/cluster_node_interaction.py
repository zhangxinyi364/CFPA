import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data

class ClusterNodeInteraction(nn.Module):
    """Node-Cluster Interaction Module Implemented Based on Cluster_GT_N2C_L"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feat_dim = args.feat_dim
        self.dropout = args.dropout
        
        # DeepSet layers are used to process node and cluster features.
        self.node_pre_deepset = self._build_deepset_layer(args.feat_dim, args.feat_dim)
        self.node_post_deepset = self._build_deepset_layer(args.feat_dim, args.feat_dim)
        self.cluster_pre_deepset = self._build_deepset_layer(args.feat_dim, args.feat_dim)
        self.cluster_post_deepset = self._build_deepset_layer(args.feat_dim, args.feat_dim)
        
        # Linear Transformation Layer
        self.linQ = nn.Linear(args.feat_dim, args.feat_dim)  # node query
        self.linK = nn.Linear(args.feat_dim, args.feat_dim)  # cluster key
        self.linV = nn.Linear(args.feat_dim, args.feat_dim)  # cluster value
        
        # Feature Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(args.feat_dim * 2, args.feat_dim),
            nn.ReLU(),
            nn.LayerNorm(args.feat_dim),
            nn.Dropout(p=args.dropout)
        )

    def _build_deepset_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=self.dropout)
        )

    def _prepare_contact_graph(self, part_positions, match_ids):
        """
        Construct the graph using the precomputed match_ids
        :param part_positions: [B, P, 3] part positions (or other features)
        :param match_ids: [B, P] part match_id
        :return: all_edges, all_edge_attrs, batch_ids
        """
        batch_size = part_positions.size(0)
        all_edges = []
        all_edge_attrs = []
        batch_ids = []

        for b in range(batch_size):
            edges_i, edges_j = [], []
            edge_weights = []
            valid_mask = match_ids[b]

            # Retrieve the quantity of parts corresponding to each match_id
            match_id_counts = {}
            for i in range(len(valid_mask)):
                if valid_mask[i]:
                    match_id = match_ids[b, i].item()  # Use .item() to obtain integer values
                    if match_id not in match_id_counts:
                        match_id_counts[match_id] = 0
                    match_id_counts[match_id] += 1

            # Iterate through all parts and join them based on match_id.
            for i in range(len(valid_mask)):
                for j in range(i + 1, len(valid_mask)):
                    if valid_mask[i] and valid_mask[j] and match_ids[b, i].item() == match_ids[b, j].item():  # .item() converts to an integer
                        edges_i.append(i)
                        edges_j.append(j)
                        
                        # Get the number of parts for the current match_id
                        match_id = match_ids[b, i].item()  # Ensure that match_id is an integer.
                        num_parts_in_bbx = match_id_counts[match_id]
                        
                        # Use the number of internal parts within the bounding box as weights
                        weight = 1 / num_parts_in_bbx
                        edge_weights.append(weight)

            if edges_i:
                edge_index = torch.stack([torch.tensor(edges_i, device=part_positions.device),
                                        torch.tensor(edges_j, device=part_positions.device)], dim=0)
                edge_attr = torch.tensor(edge_weights, device=part_positions.device)
            else:
                # Handling cases without edges
                edge_index = torch.zeros((2, 1), device=part_positions.device, dtype=torch.long)
                edge_attr = torch.zeros(1, device=part_positions.device)

            all_edges.append(edge_index)
            all_edge_attrs.append(edge_attr)
            batch_ids.extend([b] * edge_index.size(1))

        return all_edges, all_edge_attrs, torch.tensor(batch_ids, device=part_positions.device)

    



    def _compute_laplacian_norm(self, edge_index, edge_attr, num_nodes):
        """Calculate Laplace normalization"""
        dev = edge_index.device
        # Calculate the degree matrix
        deg = torch.zeros(num_nodes, device=dev)
        deg.index_add_(0, edge_index[0], edge_attr)
        
        # Calculate the inverse square root of the degree matrix
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalized Edge Weights
        norm = deg_inv_sqrt[edge_index[0]] * edge_attr * deg_inv_sqrt[edge_index[1]]
        return norm

    def forward(self, node_feats, cluster_feats, contact_points, part_valids):
        """
        Args:
            node_feats: [B, P, D] node features
            cluster_feats: [B, S, D] cluster features 
            contact_points: [B, P, P, 4] contact points information
            part_valids: [B, P] Component Validity Mask
        """
        batch_size, num_parts, feat_dim = node_feats.size()
        _, num_clusters, _ = cluster_feats.size()
        
        # 1. Feature transformation
        node_feats = self.node_pre_deepset(node_feats)  # [B, P, D]
        cluster_feats = self.cluster_pre_deepset(cluster_feats)  # [B, S, D]
        
        # 2. Cluster-to-Node Attention
        Q = self.linQ(node_feats)      # [B, P, D]
        K = self.linK(cluster_feats)   # [B, S, D]
        V = self.linV(cluster_feats)   # [B, S, D]
        
        # calculate attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, P, S]
        attention_scores = attention_scores / (feat_dim ** 0.5)
        
        # Apply softmax to the cluster dimension
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, P, S]
        
        # Calculate weighted cluster features
        node_attended = torch.bmm(attention_weights, V)  # [B, P, D]
        
        # 3. Constructing Graph Structures
        edges, edge_attrs, batch_ids = self._prepare_contact_graph(contact_points, part_valids)
        
        # 4. Image dissemination
        node_gathered_list = []
        for b in range(batch_size):
            if edges[b].size(1) > 0:
                norm = self._compute_laplacian_norm(edges[b], edge_attrs[b], num_parts)
                
                # Communication Characteristics
                cur_gathered = scatter(
                    norm.unsqueeze(-1) * node_attended[b][edges[b][0]],  # source nodes
                    edges[b][1],  # target nodes
                    dim=0,
                    dim_size=num_parts,  # Ensure the output dimensions are correct.
                    reduce='sum'
                )
            else:
                cur_gathered = node_attended[b]  # If there are no edges, retain the original features.
            
            node_gathered_list.append(cur_gathered)
        
        node_gathered = torch.stack(node_gathered_list, dim=0)  # [B, P, D]
        
        # 5. Post-processing
        node_gathered = self.node_post_deepset(node_gathered)  # [B, P, D]
        
        # 6. feature fusion
        output = self.fusion(torch.cat([node_feats, node_gathered], dim=-1))  # [B, P, D]
        
        return output