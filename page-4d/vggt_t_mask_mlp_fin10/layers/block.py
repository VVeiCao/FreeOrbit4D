# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings
import pdb
import torch
from torch import nn, Tensor
import math

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
import torch.nn.functional as F


XFORMERS_AVAILABLE = False

class SpatialMaskHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj0 = nn.Linear(d, d)      # 可选
        self.head0 = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, groups=d),    # depthwise
            nn.GELU(), nn.Conv2d(d, 1, 1))
        alpha_init: float = 3.0
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init).log())

    def forward(self, x, patch_start, H, W):  # (B,P,d)
        B, S, P, d = x.shape
        x = x.view(B*S, P, d)[:, patch_start:, :]
        
        h0 = self.proj0(x).transpose(1,2).reshape(B*S, d, H, W)
        m0 = torch.sigmoid(self.head0(h0))      # (B*S, 1, H,W)
        alpha = F.softplus(self.alpha_logit) + 1e-6
        input_mask0 = - alpha * (1 - m0).view(B, S, H*W)

        padding_mask = torch.zeros(B, S, patch_start).to(x.device)
        input_mask0 = torch.cat([padding_mask, input_mask0], dim=-1).view(B, S*P)
    
        whole_mask = torch.zeros(B, P*S, P*S).to(x.device)
        whole_mask[:, 0, :] = input_mask0
        return whole_mask
    
class SpatialMaskHead_IMP(nn.Module):
    def __init__(self, d, alpha_init: float = 1.0, tau_init: float = 2.0):
        super().__init__()
        self.head0 = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, groups=d),  # depthwise
            nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d),  # depthwise
            nn.GELU(),
            nn.Conv2d(d, 1, 1))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init).log())
        self.tau_logit   = nn.Parameter(torch.tensor(tau_init).log())
        self.soft_mask_bias = -0.5
        self.eps = 1e-6
    def forward(self, x, patch_start, H, W):  # x: (B,S,P,d)
        B, S, P, d = x.shape
        assert (P - patch_start) == (H * W), \
            f"H*W({H*W}) != P - patch_start({P - patch_start})"
        xs = x.view(B * S, P, d)[:, patch_start:, :]            # (B*S, H*W, d)
        h0 = xs.transpose(1, 2).reshape(B * S, d, H, W)
        m_logit = self.head0(h0)                                # (B*S,1,H,W)
        
        tau   = F.softplus(self.tau_logit) +  self.eps
        tau   = torch.clamp(tau, min=1e-3)  
        alpha = F.softplus(self.alpha_logit)  + self.eps
        suppress = torch.sigmoid(-m_logit / tau)                # (B*S,1,H,W), 越大越不抑制
        suppress = suppress.view(B, S, H * W)                   # ✅ 关键修正 ①

        key_vis = x.new_ones(B, S, P)
        key_vis[:, :, patch_start:] = 1.0 - torch.clamp(alpha * suppress, 0.0, 1.0)
        key_bias_1d = (1.0 - key_vis.view(B, S * P)) * self.soft_mask_bias   # (B, S*P)

        cam_row_mask = x.new_zeros(B, S, P, dtype=torch.bool)
        cam_row_mask[:, :, :patch_start] = True
        cam_row_mask = cam_row_mask.view(B, S*P)
        return key_bias_1d, cam_row_mask


class _TinyUNet(nn.Module):
    """轻量 U-Net，用于从 patch 特征预测可见性图 vis ∈ [0,1]."""
    def __init__(self, c_in, base=128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(c_in, base, 3, padding=1, groups=1),
            nn.GELU(),
            nn.GroupNorm(8, base),)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1, groups=1),
            nn.GELU(),
            nn.GroupNorm(8, base * 2),)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.out = nn.Conv2d(base, 1, 1)
        # 温控参数
        self.alpha_logit = nn.Parameter(torch.tensor(0.2).log())
        self.tau_logit   = nn.Parameter(torch.tensor(6.0).log())
    def forward(self, x_2d):  # x_2d: (B*S, C, H, W)
        e1 = self.enc1(x_2d)
        e2 = self.enc2(self.down1(e1))
        up = self.up1(e2) + e1
        m  = self.out(up)  # (B*S,1,H,W)
        eps = 1e-6
        tau   = F.softplus(self.tau_logit) + eps
        alpha = F.softplus(self.alpha_logit) + eps
        suppress = torch.sigmoid(-m / tau)                # 越大越抑制
        vis = 1.0 - torch.clamp(alpha * suppress, 0.0, 1.0)  # 可见性 ∈ [0,1]
        return vis  # (B*S,1,H,W)

class SpatialMaskHead_IMP_UNET(nn.Module):
    def __init__(self, d, base=128, max_bias=0.2):
        super().__init__()
        self.proj = nn.Linear(d, d)  # 将 token 投影到卷积空间
        self.unet = _TinyUNet(c_in=d, base=base)
        self.max_bias = max_bias     # bias 幅度上限（负值用于抑制）
    def forward(self, x, patch_start, H, W):
        """
        x: (B, S, P, d)
        patch_start: 特殊 token 起始位置（相机 + register 的数量）
        H, W: patch 网格尺寸（与 P - patch_start 对齐）
        """
        B, S, P, d = x.shape
        assert (P - patch_start) == (H * W), \
            f"H*W({H*W}) != P - patch_start({P - patch_start})"
        # 取 patch tokens -> (B*S, H*W, d) -> proj -> (B*S, d, H, W)
        xs = x.view(B * S, P, d)[:, patch_start:, :]                  # (B*S, H*W, d)
        x2d = self.proj(xs).transpose(1, 2).reshape(B * S, d, H, W)  # (B*S, d, H, W)
        # U-Net 预测可见性 vis -> (B,S,H*W)
        vis = self.unet(x2d).view(B, S, H * W)  # ∈ [0,1]
        # 拼回到 S*P 上：特殊 token 部分置 1（不抑制），patch 部分用 vis
        key_vis = x.new_ones(B, S, P)
        key_vis[:, :, patch_start:] = vis
        # 生成行 bias（负值=抑制），只在相机/注册“行”生效
        key_bias_1d = -(1.0 - key_vis.view(B, S * P)) * self.max_bias  # (B, S*P)
        # 可选：每行零均值，缓和对数值的整体漂移（更稳）
        key_bias_1d = key_bias_1d - key_bias_1d.mean(dim=-1, keepdim=True)
        # cam_row_mask：仅相机/注册行为 True
        cam_row_mask = x.new_zeros(B, S, P, dtype=torch.bool)
        cam_row_mask[:, :, :patch_start] = True
        cam_row_mask = cam_row_mask.view(B, S * P)
        return key_bias_1d, cam_row_mask

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, temporal_features = None, S=None, P=None, attn_mask=None, attn_value=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, temporal_features = None, S=None, P=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, temporal_features=temporal_features, S=S, P=P, attn_mask=attn_mask, attn_value=attn_value))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, temporal_features=temporal_features, S=S, P=P, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, temporal_features=temporal_features, S=S, P=P))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, temporal_features=temporal_features, S=S, P=P)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
