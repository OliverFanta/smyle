# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
import kornia

# Removed import for voxel_pooling as it's no longer used

__all__ = ['LSSFPN']

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.depth_se = nn.Identity()  # Placeholder for depth feature modulation
        self.context_se = SELayer(mid_channels)  # Camera-aware context features
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.reduce_conv(x)
        context = self.context_conv(x)
        depth = self.depth_se(x)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf):
        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def create_frustum(self):
        """Generate frustum"""
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat):
        """Transfer points from camera coord to ego coord."""
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum
        points = points.repeat(batch_size, num_cams, 1, 1, 1, 1).unsqueeze(-1)
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)
        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        return points.squeeze(-1)

    def forward(self, img, intrin_mat, sensor2ego_mat):
        img_feat = self.img_neck(self.img_backbone(img))[-1]
        depth_feat = self.depth_net(img_feat)
        geom = self.get_geometry(sensor2ego_mat, intrin_mat)
        return depth_feat, geom
