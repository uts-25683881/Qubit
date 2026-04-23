from __future__ import annotations

import torch
import torch.nn as nn


NUM_JOINTS = 33

# MediaPipe Pose edges (joint index pairs).
MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
    (27, 31), (28, 32),
]


def build_normalised_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """
    Build a symmetric normalised adjacency matrix A_hat = D^-1/2 A D^-1/2.
    """
    a = torch.zeros((num_joints, num_joints), dtype=torch.float32)

    # Add graph edges as undirected links.
    for i, j in MEDIAPIPE_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0

    # Add self-loops.
    a += torch.eye(num_joints, dtype=torch.float32)

    degree = torch.sum(a, dim=1)
    d_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    return d_inv_sqrt @ a @ d_inv_sqrt


class STGCNBlock(nn.Module):
    """
    Spatial graph convolution + temporal convolution block.
    Input shape: [N, C, T, V]
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 9, dropout: float = 0.1):
        super().__init__()
        pad = (temporal_kernel - 1) // 2

        self.spatial_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # Spatial graph aggregation on joint axis V.
        # x: [N, C, T, V], a_hat: [V, V]
        x_spatial = torch.einsum("nctv,vw->nctw", x, a_hat)
        x_spatial = self.spatial_proj(x_spatial)

        x_temporal = self.temporal(x_spatial)
        x_res = self.residual(x)
        return self.relu(x_temporal + x_res)


class STGCNClassifier(nn.Module):
    """
    Lightweight ST-GCN classifier for single-person skeleton action recognition.
    Expects input shape [N, C, T, V, M], where M=1 in this project.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_joints: int = NUM_JOINTS,
        base_channels: int = 32,
    ):
        super().__init__()
        self.register_buffer("a_hat", build_normalised_adjacency(num_joints))

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Lightweight backbone for faster CPU training/inference.
        self.block1 = STGCNBlock(in_channels, c1)
        self.block2 = STGCNBlock(c1, c2)
        self.block3 = STGCNBlock(c2, c3)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V, M]
        n, c, t, v, m = x.shape
        if m != 1:
            # Merge multiple persons by mean if needed.
            x = x.mean(dim=-1, keepdim=True)

        # Remove person axis -> [N, C, T, V]
        x = x[..., 0]

        # Data batch norm over C*V for each time step.
        x = x.permute(0, 2, 3, 1).contiguous()  # [N, T, V, C]
        x = x.view(n, t, v * c)
        x = x.permute(0, 2, 1).contiguous()     # [N, C*V, T]
        x = self.data_bn(x)
        x = x.permute(0, 2, 1).contiguous().view(n, t, v, c)
        x = x.permute(0, 3, 1, 2).contiguous()  # [N, C, T, V]

        x = self.block1(x, self.a_hat)
        x = self.block2(x, self.a_hat)
        x = self.block3(x, self.a_hat)

        # Global average pooling over time and joints.
        x = x.mean(dim=(2, 3))  # [N, C_last]
        return self.head(x)
