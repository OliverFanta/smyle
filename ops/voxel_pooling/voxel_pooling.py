
# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

class VoxelPooling(Function):
    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor, voxel_num: torch.Tensor) -> torch.Tensor:
        """
        A PyTorch-based fallback implementation of voxel pooling.
        """
        batch_size = input_features.size(0)
        num_channels = input_features.size(2)
        output_features = torch.zeros(
            batch_size, voxel_num[1], voxel_num[0], num_channels, device=input_features.device
        )
        
        # Discretize geom_xyz to indices within the voxel grid
        indices = geom_xyz.long()
        mask = (
            (indices[..., 0] >= 0) & (indices[..., 0] < voxel_num[0]) &
            (indices[..., 1] >= 0) & (indices[..., 1] < voxel_num[1])
        )
        
        for b in range(batch_size):
            batch_indices = indices[b][mask[b]]
            batch_features = input_features[b][mask[b]]
            for idx, feat in zip(batch_indices, batch_features):
                output_features[b, idx[1], idx[0]] += feat  # Sum pooling

        ctx.save_for_backward(geom_xyz, input_features, voxel_num)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        # Dummy backward implementation
        grad_input_features = None
        return None, grad_input_features, None


voxel_pooling = VoxelPooling.apply
