# --- File: netvlad_pp.py (new file) ---

import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLADPlusPlus(nn.Module):
    def __init__(self, feature_dim, num_clusters):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        # Learnable cluster centers
        self.centers = nn.Parameter(torch.rand(num_clusters, feature_dim))
        # Additional parameters for NetVLAD++
        self.conv = nn.Conv1d(feature_dim, num_clusters, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x is assumed to be shape (B, T, D) 
        i.e., batch_size x clip_len x feature_dim
        """
        # (B, D, T) for 1D conv
        x_perm = x.permute(0, 2, 1)
        # Compute soft-assignment scores with conv
        assignment = self.conv(x_perm)  # (B, num_clusters, T)
        assignment = F.softmax(assignment, dim=1)  # cluster assignment along the cluster dimension

        # Now we compute the residuals to cluster centers
        x_expand = x.unsqueeze(1)  # (B, 1, T, D)
        c_expand = self.centers.unsqueeze(0).unsqueeze(2)  # (1, num_clusters, 1, D)

        # broadcast to (B, num_clusters, T, D)
        residuals = x_expand - c_expand
        # Weighted by assignment (B, num_clusters, T, D)
        weighted_res = residuals * assignment.unsqueeze(-1)

        # Summation over time dimension T -> (B, num_clusters, D)
        vlad = weighted_res.sum(dim=2)
        # L2 normalize each cluster descriptor
        vlad = F.normalize(vlad, p=2, dim=-1)
        # Flatten to get final vector: (B, num_clusters * D)
        vlad = vlad.view(x.size(0), -1)
        # In practice, might do an additional normalization step
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class NetVLADpp(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        dim: int,
        downsample: int = 1,
        upsample: bool = False
    ):
        """
        Args:
          num_clusters: K
          dim: feature‐dimensionality D
          downsample: temporal factor to reduce by (1=no downsample)
          upsample: whether to interpolate back to original T
        """
        super().__init__()
        self.K = num_clusters
        self.D = dim
        self.downsample = downsample
        self.upsample = upsample

        # cluster centers (K×D)
        self.centers = nn.Parameter(torch.randn(self.K, self.D))
        # 1×1 conv for soft‐assignment
        self.conv = nn.Conv1d(self.D, self.K, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        returns:
          if upsample=False → (B, T//d, K*D)
          if upsample=True  → (B, T,     K*D)
        """
        B, T, D = x.shape
        # (B, D, T)
        x_perm = x.permute(0, 2, 1)

        # 1) downsample in time
        if self.downsample > 1:
            x_ds = F.avg_pool1d(
                x_perm,
                kernel_size=self.downsample,
                stride=self.downsample,
                padding=0
            )
        else:
            x_ds = x_perm

        T_ds = x_ds.size(-1)  # new temporal length

        # 2) soft‐assignment on the downsampled sequence
        assignment = self.conv(x_ds)            # (B, K, T_ds)
        assignment = F.softmax(assignment, dim=1)

        # 3) compute residuals against each cluster center
        #    x back to (B, T_ds, D)
        x_seg = x_ds.permute(0, 2, 1)
        #    expand dims to (B, 1, T_ds, D) and (1, K, 1, D)
        x_e = x_seg.unsqueeze(1)
        c_e = self.centers.unsqueeze(0).unsqueeze(2)
        #    residuals: (B, K, T_ds, D)
        residuals = x_e - c_e
        #    weight them by assignment → (B, K, T_ds, D)
        weighted = residuals * assignment.unsqueeze(-1)

        # 4) **keep** the time‐axis so we get one VLAD per segment
        #    reorder to (B, T_ds, K, D)
        v = weighted.permute(0, 2, 1, 3).contiguous()
        #    L2‐normalize within each cluster‐feature vector
        v = F.normalize(v, p=2, dim=-1)
        #    flatten cluster+feat dims → (B, T_ds, K*D)
        v = v.view(B, T_ds, self.K * self.D)

        # 5) optionally upsample back to the original T
        if self.upsample and T_ds != T:
            # permute to (B, K*D, T_ds) for interpolate
            v = v.permute(0, 2, 1)
            v = F.interpolate(
                v,
                size=T,
                mode='linear',
                align_corners=False
            )
            v = v.permute(0, 2, 1)  # → (B, T, K*D)

        return v
