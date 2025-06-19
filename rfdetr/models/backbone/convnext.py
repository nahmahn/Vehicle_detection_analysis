import torch
import torch.nn as nn
import timm

size_to_timm_name = {
    "tiny": "convnext_tiny",
    "small": "convnext_small",
    "base": "convnext_base",
    "large": "convnext_large",
} #Use tiny one.

class ConvNeXt(nn.Module):
    def __init__(self, size="tiny", out_feature_indexes=[0, 1, 2, 3], pretrained=True, **kwargs):
        super().__init__()

        if size not in size_to_timm_name:
            raise ValueError(f"Unsupported ConvNeXt size: {size}. Available sizes: {list(size_to_timm_name.keys())}")

        timm_model_name = size_to_timm_name[size]
        self.encoder = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_feature_indexes,
        )
        self._out_feature_channels = [info['num_chs'] for info in self.encoder.feature_info]
        self._export = False

    def export(self):
        if self._export:
            return
        self._export = True
        print("ConvNeXt backbone requires no special export steps.")

    def forward(self, x):
        return self.encoder(x)