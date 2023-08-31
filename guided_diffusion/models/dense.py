from abc import abstractmethod

import math
from re import S
from time import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .unet import TimestepBlock, EncoderUNetModel, UNetModel
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class DenseResBlock(nn.Module):
    """Fully-connected residual block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.model_channels = model_channels
        self.resblock = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            FeaturewiseAffine(),
            nn.SiLU(),
            linear(self.out_channels, self.out_channels),
            nn.LayerNorm(self.out_channels),
            FeaturewiseAffine(),
            nn.SiLU(),
            linear(self.out_channels, self.out_channels),
        )

    def forward(self, inputs, scale, shift):
        """forward fn"""
        x = inputs
        for i in range(len(self.resblock)):
            if type(self.resblock[i]) is FeaturewiseAffine:
                x = self.resblock[i](x, scale, shift)
            else:
                x = self.resblock[i](x)

        output = x
        shortcut = inputs 
        if x.shape[-1] != self.out_channels:
            shortcut = linear(x)

        return output + shortcut

class FeaturewiseAffine(nn.Module):
    """Feature-wise affine layer."""
    def forward(self, x, scale, shift):
        return scale * x + shift
    
class DenseDDPM(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, in_channels=159, num_layers=3, model_channels=2048, use_checkpoint=False):
        super().__init__()
        self.in_layers = linear(in_channels, model_channels)
        self.use_checkpoint = use_checkpoint

        self.mid_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.mid_layers.append(nn.Sequential(*[
                DenseFiLM(128, model_channels),
                DenseResBlock(in_channels, model_channels)
            ]))
        
        self.out_layers = nn.Sequential(
            nn.LayerNorm(model_channels),
            linear(model_channels, in_channels)
        )
        
    def forward(self, inputs, t, **kwargs):
        x = inputs
        x = self.in_layers(x)

        for i in range(len(self.mid_layers)):
            scale, shift = self.mid_layers[i][0](t)    # DenseFiLM
            x = self.mid_layers[i][1](x, scale=scale, shift=shift)    # ResBlock
        out = self.out_layers(x)
        return {'output':out}

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.model_channels = model_channels 
        self.out_channels = out_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim), 
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.scale_layer = linear(time_embed_dim, self.out_channels)
        self.shift_layer = linear(time_embed_dim, self.out_channels)

    def forward(self, t):
        # position.shape = (batch_size, 1)
        # embedding_channels.shape, out_channels.shape = (), ()
        # print(t.shape)
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        scale = self.scale_layer(emb)
        shift = self.shift_layer(emb)
        return scale, shift

class AutoEncoderDPM(nn.Module):
    '''
    P'ta's architecture => https://arxiv.org/pdf/2111.15640.pdf
    '''
    def __init__(self, in_channels, num_layers, out_channels, model_channels, use_checkpoint):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.skip_connection = list(range(0, self.num_layers))

        self.in_layers = linear(in_channels, model_channels)

        self.mid_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            if i == 0:
                self.mid_layers.append(nn.Sequential(*[
                    DenseFiLM(128, model_channels),
                    AutoEncoderResBlock(model_channels, model_channels)
                ]))
            else:
                self.mid_layers.append(nn.Sequential(*[
                    DenseFiLM(128, model_channels+in_channels),
                    AutoEncoderResBlock(model_channels+in_channels, model_channels)
                ]))

        self.out_layers = nn.Sequential(
                nn.LayerNorm(self.model_channels+in_channels),
                linear(self.model_channels+in_channels, self.out_channels),
            )
    
    def forward(self, inputs, t):
        x = inputs
        h = self.in_layers(x)

        for i in range(len(self.mid_layers)):
            scale, shift = self.mid_layers[i][0](t)    # DenseFiLM
            h = self.mid_layers[i][1](h, scale=scale, shift=shift)    # ResBlock
            if i in self.skip_connection:
                h = th.cat((h, inputs), dim=1)

        out = self.out_layers(h)
        return {'output':out}

class AutoEncoderResBlock(nn.Module):
    '''
    Encoder block consisted of In -> MLP -> FiLM -> LayerNorm -> concat(Out, In)
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.resblock = nn.Sequential(
            nn.SiLU(),
            FeaturewiseAffine(),
            nn.LayerNorm(self.in_channels),
            linear(self.in_channels, self.out_channels),
        )

    def forward(self, inputs, scale, shift):
        x = inputs
        for i in range(len(self.resblock)):
            if type(self.resblock[i]) is FeaturewiseAffine:
                x = self.resblock[i](x, scale, shift)
            else:
                x = self.resblock[i](x)

        output = x

        return output

class DenseDDPMCond(DenseDDPM):
    def __init__(self, 
        encoder,
        in_channels=159, 
        num_layers=3, 
        model_channels=2048, 
        use_checkpoint=False,
        ):

        DenseDDPM.__init__(
            self,
            in_channels=in_channels,
            model_channels=model_channels,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint
        )

        self.encoder = encoder

    def forward(self, inputs, t, **kwargs):
        '''
        :params x: input parameters B x #dims; e.g. #dims of DECA = 159
        :params cond: conditioning the network. 
            - In this case, we used image as input to our encoderUNet to output the latent
            and used this to condition the DenseDDPM. B x 3 x H x W
        '''
        x = inputs

        # Conditioning from an image
        cond = kwargs['image']
        cond = self.encoder(x=cond.type_as(x), timesteps=t)

        # DenseDDPM
        x = self.in_layers(x) * cond

        for i in range(len(self.mid_layers)):
            scale, shift = self.mid_layers[i][0](t)    # DenseFiLM
            x = self.mid_layers[i][1](x, scale=scale, shift=shift) * cond    # ResBlock
        out = self.out_layers(x)

        return {'output':out}