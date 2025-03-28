"""
LorentzNet: Lorentz-Equivariant Graph Neural Network for Jet Tagging

This module implements the LorentzNet architecture, a GNN that preserves Lorentz symmetry.
Used for particle physics tasks like jet classification.

Adapted from:
https://github.com/sdogsq/LorentzNet-release

Based on the paper:
"An Efficient Lorentz Equivariant Graph Neural Network for Jet Tagging"
https://arxiv.org/abs/2201.08187
"""

import torch
from torch import nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def normsq4(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


class LGEB(nn.Module):
    """
    Lorentz Graph Embedding Block (LGEB): a Lorentz-invariant message-passing layer.

    Args:
        n_input (int): Input feature dimension.
        n_output (int): Output dimension.
        n_hidden (int): Hidden layer size.
        n_node_attr (int): Number of extra scalar features.
        dropout (float): Dropout rate.
        c_weight (float): Weight for coordinate update.
        last_layer (bool): If True, disables coordinate updates.
    """
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super().__init__()
        self.c_weight = c_weight
        self.last_layer = last_layer

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + 2, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

        if not last_layer:
            self.phi_x = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1, bias=False)
            )
            torch.nn.init.xavier_uniform_(self.phi_x[2].weight, gain=0.001)

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.message_model(h[i], h[j], norms, dots)
        if not self.last_layer:
            x = self.coord_model(x, edges, x_diff, m)
        h = self.update_model(h, edges, m, node_attr)
        return h, x, m

    def message_model(self, hi, hj, norms, dots):
        msg = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(msg)
        w = self.phi_m(out)
        return out * w

    def update_model(self, h, edges, m, node_attr):
        i, _ = edges
        agg = unsorted_segment_sum(m, i, h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        return h + self.phi_h(agg)

    def coord_model(self, x, edges, x_diff, m):
        i, _ = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, x.size(0))
        return x + agg * self.c_weight

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        return psi(norms), psi(dots), x_diff


class LorentzNet(nn.Module):
    """
    LorentzNet model: stacked LGEB blocks + classifier for jet tagging.

    Args:
        n_scalar (int): Number of input scalar features.
        n_hidden (int): Hidden feature size.
        n_class (int): Number of output classes.
        n_layers (int): Number of LGEB layers.
        c_weight (float): Coordinate update weight.
        dropout (float): Dropout rate.
    """
    def __init__(self, n_scalar, n_hidden, n_class=10, n_layers=6, c_weight=1e-3, dropout=0.):
        super().__init__()
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([
            LGEB(n_hidden, n_hidden, n_hidden, n_node_attr=n_scalar,
                 dropout=dropout, c_weight=c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class)
        )

    def forward(self, scalars, p4s, edges, atom_mask, edge_mask, n_nodes):
        """
        Args:
            scalars: (B*N, n_scalar)
            p4s:     (B*N, 4)
            edges:   (2, E)
            atom_mask: (B, N) or (B, N, 1)
            n_nodes: int, number of nodes per event

        Returns:
            logits: (B, n_class)
        """
        h = self.embedding(scalars)  # (B*N, hidden)
        x = p4s
        node_attr = scalars

        for layer in self.LGEBs:
            h, x, _ = layer(h, x, edges, node_attr)

        # Reshape and mask
        h = h.view(-1, n_nodes, h.size(-1))  # (B, N, hidden)

        if atom_mask.dim() == 2:
            atom_mask = atom_mask.unsqueeze(-1)  # (B, N, 1)

        h_masked = h * atom_mask  # (B, N, hidden)
        h_sum = h_masked.sum(dim=1)  # (B, hidden)
        norm = atom_mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        h_mean = h_sum / norm  # (B, hidden)

        return self.graph_dec(h_mean)  # (B, n_class)

