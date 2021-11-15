import torch
from torch import nn

import timm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class VideoLightFormer(nn.Module):
  def __init__(self, 
               backbone_model_name,
               num_classes):
    super().__init__()
    
    self.f = nn.Sequential()
    self.g = None
    self.h = None
    self.mlp = None
  
  def forward(self, x):
    return self.mlp(self.h(self.g(self.f(x))))
  
