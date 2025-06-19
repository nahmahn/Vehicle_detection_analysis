# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Dict, List

import torch
from torch import nn

from rfdetr.util.misc import NestedTensor
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.models.backbone.backbone import *
from typing import Callable
# Use a direct relative import to the new, unified backbone.py
from .backbone import Backbone

# This helper class needs to be defined here for build_backbone to work
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        # The backbone (which includes the projector) returns a list of NestedTensors
        xs = self[0](tensor_list)
        out = []
        pos = []
        for x in xs:
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

    # ... (export methods can be kept if needed, but are omitted for clarity) ...


# REPLACE THE EXISTING 'build_backbone' FUNCTION WITH THIS:

def build_backbone(args):
    """
    Builds the main backbone module and the position encoding.
    """
    position_embedding = build_position_encoding(args.hidden_dim, args.position_embedding)
    
    # --- This is the fix: The function now takes a single `args` object ---
    backbone = Backbone(
        name=args.encoder,
        out_channels=args.hidden_dim,
        projector_scale=args.projector_scale,
        out_feature_indexes=args.out_feature_indexes,
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        freeze_encoder=getattr(args, 'freeze_encoder', False),
        layer_norm=getattr(args, 'layer_norm', False),
        rms_norm=getattr(args, 'rms_norm', False),
        load_dinov2_weights=getattr(args, 'load_dinov2_weights', True if args.pretrain_weights is None else False),
    )

    model = Joiner(backbone, position_embedding)
    
    # Set the num_channels attribute on the Joiner, which is used by the main DETR model
    model.num_channels = args.hidden_dim
    
    return model

# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)
#         self._export = False

#     def forward(self, tensor_list: NestedTensor):
#         """ """
#         x = self[0](tensor_list)
#         pos = []
#         for x_ in x:
#             pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
#         return x, pos

#     def export(self):
#         self._export = True
#         self._forward_origin = self.forward
#         self.forward = self.forward_export
#         for name, m in self.named_modules():
#             if (
#                 hasattr(m, "export")
#                 and isinstance(m.export, Callable)
#                 and hasattr(m, "_export")
#                 and not m._export
#             ):
#                 m.export()

#     def forward_export(self, inputs: torch.Tensor):
#         feats, masks = self[0](inputs)
#         poss = []
#         for feat, mask in zip(feats, masks):
#             poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
#         return feats, None, poss


# def build_backbone(
#     encoder,
#     vit_encoder_num_layers,
#     pretrained_encoder,
#     window_block_indexes,
#     drop_path,
#     out_channels,
#     out_feature_indexes,
#     projector_scale,
#     use_cls_token,
#     hidden_dim,
#     position_embedding,
#     freeze_encoder,
#     layer_norm,
#     target_shape,
#     rms_norm,
#     backbone_lora,
#     force_no_pretrain,
#     gradient_checkpointing,
#     load_dinov2_weights,
# ):
#     """
#     Useful args:
#         - encoder: encoder name
#         - lr_encoder:
#         - dilation
#         - use_checkpoint: for swin only for now

#     """
#     position_embedding = build_position_encoding(hidden_dim, position_embedding)

#     backbone = Backbone(
#         encoder,
#         pretrained_encoder,
#         window_block_indexes=window_block_indexes,
#         drop_path=drop_path,
#         out_channels=out_channels,
#         out_feature_indexes=out_feature_indexes,
#         projector_scale=projector_scale,
#         use_cls_token=use_cls_token,
#         layer_norm=layer_norm,
#         freeze_encoder=freeze_encoder,
#         target_shape=target_shape,
#         rms_norm=rms_norm,
#         backbone_lora=backbone_lora,
#         gradient_checkpointing=gradient_checkpointing,
#         load_dinov2_weights=load_dinov2_weights,
#     )

#     model = Joiner(backbone, position_embedding)
#     return model
