# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ... (other copyrights) ...
# ------------------------------------------------------------------------

# --- This is the original DINOv2-only version, preserved as requested ---
# # ------------------------------------------------------------------------
# # RF-DETR
# # Copyright (c) 2025 Roboflow. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# # Copyright (c) 2024 Baidu. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# # Copyright (c) 2021 Microsoft. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Copied from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # ------------------------------------------------------------------------

# """
# Backbone modules.
# """
# from functools import partial
# import torch
# import torch.nn.functional as F
# from torch import nn

# from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
# from peft import LoraConfig, get_peft_model, PeftModel

# from rfdetr.util.misc import NestedTensor, is_main_process

# from rfdetr.models.backbone.base import BackboneBase
# from rfdetr.models.backbone.projector import MultiScaleProjector
# from rfdetr.models.backbone.dinov2 import DinoV2

# __all__ = ["Backbone"]


# class Backbone(BackboneBase):
#     """backbone."""
#     def __init__(self,
#                  name: str,
#                  pretrained_encoder: str=None,
#                  window_block_indexes: list=None,
#                  drop_path=0.0,
#                  out_channels=256,
#                  out_feature_indexes: list=None,
#                  projector_scale: list=None,
#                  use_cls_token: bool = False,
#                  freeze_encoder: bool = False,
#                  layer_norm: bool = False,
#                  target_shape: tuple[int, int] = (640, 640),
#                  rms_norm: bool = False,
#                  backbone_lora: bool = False,
#                  gradient_checkpointing: bool = False,
#                  load_dinov2_weights: bool = True,
#                  ):
#         super().__init__()
#         # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
#         # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
#         # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
#         # the last part of the name should be the size
#         # and the start should be dinov2
#         name_parts = name.split("_")
#         assert name_parts[0] == "dinov2"
#         size = name_parts[-1]
#         use_registers = False
#         if "registers" in name_parts:
#             use_registers = True
#             name_parts.remove("registers")
#         use_windowed_attn = False
#         if "windowed" in name_parts:
#             use_windowed_attn = True
#             name_parts.remove("windowed")
#         assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
#         self.encoder = DinoV2(
#             size=name_parts[-1],
#             out_feature_indexes=out_feature_indexes,
#             shape=target_shape,
#             use_registers=use_registers,
#             use_windowed_attn=use_windowed_attn,
#             gradient_checkpointing=gradient_checkpointing,
#             load_dinov2_weights=load_dinov2_weights,
#         )
#         # build encoder + projector as backbone module
#         if freeze_encoder:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         self.projector_scale = projector_scale
#         assert len(self.projector_scale) > 0
#         # x[0]
#         assert (
#             sorted(self.projector_scale) == self.projector_scale
#         ), "only support projector scale P3/P4/P5/P6 in ascending order."
#         level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
#         scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

#         self.projector = MultiScaleProjector(
#             in_channels=self.encoder._out_feature_channels,
#             out_channels=out_channels,
#             scale_factors=scale_factors,
#             layer_norm=layer_norm,
#             rms_norm=rms_norm,
#         )

#         self._export = False

#     def export(self):
#         self._export = True
#         self._forward_origin = self.forward
#         self.forward = self.forward_export

#         if isinstance(self.encoder, PeftModel):
#             print("Merging and unloading LoRA weights")
#             self.encoder.merge_and_unload()

#     def forward(self, tensor_list: NestedTensor):
#         """ """
#         # (H, W, B, C)
#         feats = self.encoder(tensor_list.tensors)
#         feats = self.projector(feats)
#         # x: [(B, C, H, W)]
#         out = []
#         for feat in feats:
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
#                 0
#             ]
#             out.append(NestedTensor(feat, mask))
#         return out

#     def forward_export(self, tensors: torch.Tensor):
#         feats = self.encoder(tensors)
#         feats = self.projector(feats)
#         out_feats = []
#         out_masks = []
#         for feat in feats:
#             # x: [(B, C, H, W)]
#             b, _, h, w = feat.shape
#             out_masks.append(
#                 torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
#             )
#             out_feats.append(feat)
#         return out_feats, out_masks

#     def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
#         num_layers = args.out_feature_indexes[-1] + 1
#         backbone_key = "backbone.0.encoder"
#         named_param_lr_pairs = {}
#         for n, p in self.named_parameters():
#             n = prefix + "." + n
#             if backbone_key in n and p.requires_grad:
#                 lr = (
#                     args.lr_encoder
#                     * get_dinov2_lr_decay_rate(
#                         n,
#                         lr_decay_rate=args.lr_vit_layer_decay,
#                         num_layers=num_layers,
#                     )
#                     * args.lr_component_decay**2
#                 )
#                 wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
#                 named_param_lr_pairs[n] = {
#                     "params": p,
#                     "lr": lr,
#                     "weight_decay": wd,
#                 }
#         return named_param_lr_pairs


# def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
#     """
#     Calculate lr decay rate for different ViT blocks.

#     Args:
#         name (string): parameter name.
#         lr_decay_rate (float): base lr decay rate.
#         num_layers (int): number of ViT blocks.
#     Returns:
#         lr decay rate for the given parameter.
#     """
#     layer_id = num_layers + 1
#     if name.startswith("backbone"):
#         if "embeddings" in name:
#             layer_id = 0
#         elif ".layer." in name and ".residual." not in name:
#             layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
#     return lr_decay_rate ** (num_layers + 1 - layer_id)

# def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
#     if (
#         ("gamma" in name)
#         or ("pos_embed" in name)
#         or ("rel_pos" in name)
#         or ("bias" in name)
#         or ("norm" in name)
#         or ("embeddings" in name)
#     ):
#         weight_decay_rate = 0.0
#     return weight_decay_rate


import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.util.misc import NestedTensor

from .base import BackboneBase
from .projector import MultiScaleProjector
from .convnext import ConvNeXt
from .dinov2 import DinoV2, get_dinov2_lr_decay_rate, get_dinov2_weight_decay_rate

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """
    Backbone module that dispatches to DINOv2 or ConvNeXt.
    It initializes the specified encoder and connects it to a MultiScaleProjector neck.
    """
    def __init__(self,
                 name: str,
                 out_channels=256,
                 projector_scale: list = None,
                 # DINOv2 specific args
                 out_feature_indexes: list = None,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 # General args
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 rms_norm: bool = False,
                 **kwargs): # Absorb unused kwargs
        super().__init__()

        # --- Backbone Dispatcher Logic ---
        if "dinov2" in name:
            name_parts = name.split("_")
            assert name_parts[0] == "dinov2"
            size = name_parts[-1]
            use_registers = "registers" in name_parts
            use_windowed_attn = "windowed" in name_parts

            self.encoder = DinoV2(
                size=size,
                out_feature_indexes=out_feature_indexes,
                use_registers=use_registers,
                use_windowed_attn=use_windowed_attn,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
            )
            print(f"Initialized DINOv2 backbone: {name}")

        elif "convnext" in name:
            name_parts = name.split("_")
            assert name_parts[0] == "convnext", "ConvNeXt name should be in the format 'convnext_<size>'"
            size = name_parts[-1]
            # ConvNeXt always outputs 4 feature maps
            convnext_out_indices = [0, 1, 2, 3]

            self.encoder = ConvNeXt(
                size=size,
                out_feature_indexes=convnext_out_indices,
                pretrained=True
            )
            print(f"Initialized ConvNeXt backbone: {name}")
        else:
            raise ValueError(f"Unsupported backbone name: '{name}'. Must contain 'dinov2' or 'convnext'.")
        # --- End of Dispatcher Logic ---

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert projector_scale is not None and len(projector_scale) > 0, "projector_scale must not be empty"

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            # Pass the scale names directly to the new projector
            scale_names=self.projector_scale,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def export(self):
        """Prepares the model for export by calling the encoder's export method."""
        if self._export: return
        self._export = True
        self.encoder.export()

    def forward(self, tensor_list: NestedTensor):
        """Standard forward pass for training."""
        # The encoder returns a list of feature maps
        feats = self.encoder(tensor_list.tensors)
        # The projector processes these feature maps
        feats = self.projector(feats)

        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out


    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        """
        Dispatcher for getting learning rate pairs.
        It calls the appropriate method based on the encoder type.
        """
        if isinstance(self.encoder, DinoV2):
            return self._get_dinov2_lr_pairs(args, prefix)
        elif isinstance(self.encoder, ConvNeXt):
            return self._get_generic_lr_pairs(args, prefix)
        else:
            raise TypeError(f"LR pair generation not implemented for encoder type: {type(self.encoder)}")

    def _get_generic_lr_pairs(self, args, prefix: str):
        """A generic learning rate configuration for CNNs like ConvNeXt."""
        named_param_lr_pairs = {}
        backbone_key = "backbone.0.encoder"

        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": args.lr_encoder,
                    "weight_decay": args.weight_decay,
                }
        return named_param_lr_pairs

    def _get_dinov2_lr_pairs(self, args, prefix: str):
        """Original DINOv2 specific layer-wise learning rate decay logic."""
        num_layers = 12 # DINOv2 base has 12 layers
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs