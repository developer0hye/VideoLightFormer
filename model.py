import torch
from torch import nn

import timm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Transformer(nn.Module):
  def __init__(self, 
               num_encoders,
               num_heads,
               mlp_ratio=4.0):
    super().__init__()
  def forward(self, x):
    return x

class VideoLightFormer(nn.Module):
  def __init__(self,
               num_classes,
               backbone_model_name="tf_efficientnet_b0_ns",
               embed_dim=256,
               num_frames=16,
               ):
    super().__init__()

    self.backbone = timm.create_model(model_name=backbone_model_name,
                                      pretrained=True,
                                      features_only=True)
    
    backbone_out_channels = self.backbone.feature_info.channels()[-1]

    self.conv = nn.Conv2d(in_channels=backbone_out_channels,
                          out_channels=256,
                          kernel_size=3,
                          stride=2)
    
    self.bn = nn.BatchNorm2d(256)
    
    self.g = Transformer(num_encoders=1, num_heads=8)
    self.h = nn.Sequential(Rearrange('(b t) n c -> b (t n) c', t=num_frames),
                           Transformer(num_encoders=4,
                                       num_heads=8))
    self.mlp = nn.Sequential(Rearrange('b tn c -> b c tn'),
                             nn.AdaptiveAvgPool1d(1),
                             nn.Flatten(start_dim=1),
                             nn.Linear(256, num_classes))
  
  def f(self, x):
    x = self.backbone.forward(x)[-1]
    x = self.conv(x)
    x = self.bn(x)

    x = rearrange(x, 'b c h w -> b (h w) c')
    return x
  
  def forward(self, x):
    return self.mlp(self.h(self.g(self.f(x))))
