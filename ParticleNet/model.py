"""

This module defines the full implementation of ParticleNet, a neural network
architecture designed for jet tagging tasks using particle cloud representations.

Main components:
- `EdgeConvBlock`: Performs edge convolutions over particle neighborhoods
- `ParticleNet`: Full architecture with edge conv blocks, optional fusion layers,
  and fully connected classifier
- Utility functions:
    - `batch_distance_matrix_general`: Computes pairwise squared Euclidean distance
    - `knn`: Advanced indexing to get k-nearest neighbor features

Expected inputs:
- points: Tensor of shape (N, P, 3) or similar (e.g., eta, phi, pt)
- features: Tensor of shape (N, P, C) with padded particle features
- mask: Optional binary mask (N, P, 1) indicating valid particles

Output:
- logits: Tensor of shape (N, num_classes)

Adapted from paper: ParticleNet: Jet Tagging via Particle Clouds
Url: https://arxiv.org/abs/1902.08570
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_distance_matrix_general(A, B):
    """
    Compute squared Euclidean distances between each point in A and B.

    Args:
        A (torch.Tensor): Shape (N, P_A, C)
        B (torch.Tensor): Shape (N, P_B, C)

    Returns:
        torch.Tensor: Distance matrix of shape (N, P_A, P_B)
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)  # (N, P_A, 1)
    r_B = torch.sum(B * B, dim=2, keepdim=True)  # (N, P_B, 1)
    m = torch.bmm(A, B.transpose(1, 2))          # (N, P_A, P_B)
    D = r_A - 2 * m + r_B.transpose(1, 2)        # (N, P_A, P_B)
    return D


def knn(topk_indices, features):
    """
    Gather features of k-nearest neighbors using advanced indexing.

    Args:
        topk_indices (torch.Tensor): Shape (N, P, K) with neighbor indices
        features (torch.Tensor): Shape (N, P, C)

    Returns:
        torch.Tensor: Shape (N, P, K, C)
    """
    N, P, K = topk_indices.shape
    batch_indices = torch.arange(N, device=features.device).view(N, 1, 1).expand(N, P, K)
    return features[batch_indices, topk_indices, :]


class EdgeConvBlock(nn.Module):
    """
    One EdgeConv block with optional batch norm and pooling (average or max).

    Args:
        K (int): Number of neighbors
        in_channels (int): Input feature dimension
        channels (list[int]): List of output channels for each Conv1x1 layer
        with_bn (bool): Whether to use batch normalization
        activation (nn.Module): Activation function
        pooling (str): 'average' or 'max' over neighbors
    """
    def __init__(self, K, in_channels, channels, with_bn=True, activation=nn.ReLU(), pooling='average'):
        super().__init__()
        self.K = K
        self.with_bn = with_bn
        self.pooling = pooling
        self.activation = activation

        in_ch = 2 * in_channels
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in channels:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not with_bn)
            self.convs.append(conv)
            if with_bn:
                self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        # Shortcut connection
        self.shortcut_conv = nn.Conv1d(in_channels, channels[-1], kernel_size=1, bias=not with_bn)
        if with_bn:
            self.shortcut_bn = nn.BatchNorm1d(channels[-1])
        self.shortcut_activation = activation

    def forward(self, points, features):
        """
        Forward pass of the EdgeConv block.

        Args:
            points (torch.Tensor): Shape (N, P, C)
            features (torch.Tensor): Shape (N, P, in_channels)

        Returns:
            torch.Tensor: Output of shape (N, P, out_channels)
        """
        N, P, C = features.shape
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = torch.topk(-D, k=self.K + 1, dim=2)
        indices = indices[:, :, 1:]  # exclude self

        knn_feats = knn(indices, features)  # (N, P, K, C)
        center_feats = features.unsqueeze(2).expand(-1, -1, self.K, -1)
        x = torch.cat([center_feats, knn_feats - center_feats], dim=-1)  # (N, P, K, 2C)
        x = x.permute(0, 3, 1, 2)  # (N, 2C, P, K)

        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if self.with_bn:
                x = self.bns[idx](x)
            x = self.activation(x)

        x = torch.mean(x, dim=3) if self.pooling == 'average' else torch.max(x, dim=3)[0]

        # Shortcut
        sc = self.shortcut_conv(features.permute(0, 2, 1))  # (N, out_ch, P)
        if self.with_bn:
            sc = self.shortcut_bn(sc)
        sc = self.shortcut_activation(sc)

        out = x + sc  # residual
        return out.permute(0, 2, 1)  # (N, P, out_ch)


class ParticleNet(nn.Module):
    """
    Full ParticleNet model: EdgeConv blocks + optional fusion + FC classifier.

    Args:
        input_dims (int): Input feature dimension per particle.
        num_classes (int): Number of output classes.
        conv_params (list): List of (K, [channels...]) tuples for EdgeConv blocks.
        fc_params (list): List of (units, dropout_rate) tuples for FC layers.
        conv_pooling (str): 'average' or 'max'
        use_fusion (bool): Whether to fuse multi-scale features
        use_fts_bn (bool): Whether to apply batch norm to input features
        use_counts (bool): Whether to divide global sum pooling by number of particles
    """
    def __init__(self, input_dims, num_classes,
                 conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                 fc_params=[(256, 0.1)],
                 conv_pooling='average', use_fusion=True, use_fts_bn=True, use_counts=True):
        super().__init__()
        self.use_fts_bn = use_fts_bn
        if use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.edge_convs = nn.ModuleList()
        for idx, (K, channels) in enumerate(conv_params):
            in_ch = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(
                EdgeConvBlock(K, in_ch, channels, with_bn=True, pooling=conv_pooling)
            )

        self.use_fusion = use_fusion
        if use_fusion:
            in_chn = sum(c[-1] for _, c in conv_params)
            out_chn = max(128, min(1024, (in_chn // 128) * 128))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU()
            )
        else:
            out_chn = conv_params[-1][1][-1]

        # Fully connected classifier
        fc_layers = []
        in_fc = out_chn
        for units, drop_rate in fc_params:
            fc_layers.append(nn.Linear(in_fc, units))
            fc_layers.append(nn.ReLU())
            if drop_rate:
                fc_layers.append(nn.Dropout(drop_rate))
            in_fc = units
        self.fc = nn.Sequential(*fc_layers)
        self.fc_final = nn.Linear(in_fc, num_classes)
        self.use_counts = use_counts

    def forward(self, points, features, mask=None):
        """
        Forward pass of ParticleNet.

        Args:
            points (torch.Tensor): (N, P, C_coords) – e.g. particle coordinates
            features (torch.Tensor): (N, P, input_dims) – padded particle features
            mask (torch.Tensor or None): (N, P, 1) – binary mask for valid particles

        Returns:
            torch.Tensor: (N, num_classes) – output logits
        """
        if mask is None:
            mask = (features.abs().sum(dim=2, keepdim=True) != 0).float()

        coord_shift = (mask == 0).float() * 1e9

        if self.use_fts_bn:
            features = self.bn_fts(features.permute(0, 2, 1)).permute(0, 2, 1) * mask

        outputs = []
        for idx, edge_conv in enumerate(self.edge_convs):
            pts_input = points if idx == 0 else features
            pts_input = pts_input + coord_shift  # disable masked entries
            features = edge_conv(pts_input, features) * mask
            if self.use_fusion:
                outputs.append(features)

        if self.use_fusion:
            fused = torch.cat(outputs, dim=2).permute(0, 2, 1)  # (N, sum_channels, P)
            features = self.fusion_block(fused).permute(0, 2, 1) * mask  # (N, P, out_chn)

        # Global pooling over particles
        if self.use_counts:
            counts = mask.sum(dim=1)  # (N, 1)
            pooled = features.sum(dim=1) / counts  # (N, out_chn)
        else:
            pooled = features.mean(dim=1)

        x = self.fc(pooled)
        return self.fc_final(x)
