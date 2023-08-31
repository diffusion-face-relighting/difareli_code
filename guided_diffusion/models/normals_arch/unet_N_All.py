from abc import abstractmethod

import math
from time import time
from cv2 import norm

import numpy as np
import torch as th
th.set_printoptions(precision=5)
import torch.nn as nn
import torch.nn.functional as F
from ..renderer import Renderer

from ...trainer_util import convert_module_to_f16, convert_module_to_f32
from ..nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from ..unet import (
    AttentionBlockNormals,
    ResBlock,
    ResBlockCondition,
    TimestepBlockCond,
    TimestepEmbedSequential, 
    TimestepEmbedSequentialCond,
    AttentionBlock,
    Upsample,
    Downsample,
)

class ResBlockNormalCond(TimestepBlockCond):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        condition_dim,
        condition_proj_dim,
        renderer,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.condition_dim = condition_dim
        self.condition_proj_dim = condition_proj_dim
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.renderer = renderer
        self.num_SH = self.renderer.num_SH 

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels + self.num_SH, 3, padding=1),
        )

        self.cond_proj_layers = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_proj_dim),
            nn.SiLU(),
            nn.Linear(self.condition_proj_dim, self.condition_proj_dim),
            nn.SiLU(),
            nn.Linear(self.condition_proj_dim, self.out_channels),
            nn.SiLU(),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * (self.out_channels + self.num_SH) if use_scale_shift_norm else self.out_channels,
            ),
        )

        shading_channels = self.renderer.shading_channels
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            normalization(self.num_SH, n_group=self.num_SH),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels + shading_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x, emb, cond):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, cond):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)

        # print("cond", cond['cond_params'].shape)
        cond_proj = self.cond_proj_layers(cond['cond_params'].type(h.dtype))
        # print("cond_proj", cond_proj.shape)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, normals_norm = self.out_layers[0], self.out_layers[1]
            out_rest = self.out_layers[2:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            # print(h.shape)
            # print(out_norm(h[:, :self.out_channels, ...]).shape)
            # print(normals_norm(h[:, self.out_channels:, ...]).shape)
            h = th.cat((
                out_norm(h[:, :self.out_channels, ...]), 
                normals_norm(h[:, self.out_channels:, ...])
            ), dim=1)
            # Apply scale_shift_norm
            h = (h * (1 + scale) + shift)
            # Apply condition
            h_cond = h[:, :self.out_channels] * cond_proj[:, :, None, None].type_as(h)
            # Apply relighting
            h_normals = self.renderer.add_SHlight(normal_images=h[:, self.out_channels:, ...], sh_coeff=cond['light'])

            h = th.cat((h_cond, h_normals), dim=1).float()
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetNormalsAll(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        conditioning=False,
        condition_dim=0,
        condition_proj_dim=0,
        all_cfg=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.conditioning = conditioning
        self.condition_dim = condition_dim
        self.condition_proj_dim = condition_proj_dim

        # Relighting
        self.num_SH = all_cfg.relighting.num_SH
        self.reduce_shading = all_cfg.relighting.reduce_shading
        self.renderer = Renderer(num_SH=self.num_SH, reduce_shading=self.reduce_shading)
        self.all_cfg = all_cfg

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        resblock_module = ResBlock if not self.conditioning else ResBlockCondition
        time_embed_seq_module = TimestepEmbedSequential if not self.conditioning else TimestepEmbedSequentialCond

        if all_cfg.relighting.arch == 'add_channels':
            resblock_normals_module = ResBlockNormalCond
        else: raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [time_embed_seq_module(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    resblock_normals_module(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=condition_dim,
                        condition_proj_dim=condition_proj_dim,
                        renderer=self.renderer,
                        apply_first=all_cfg.relighting.apply_first,
                        use_conv=True,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlockNormals(
                            ch,
                            norm_type='GroupNorm',
                            shading_channels=self.renderer.shading_channels,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(time_embed_seq_module(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    time_embed_seq_module(
                        resblock_normals_module(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            condition_dim=condition_dim,
                            condition_proj_dim=condition_proj_dim,
                            renderer=self.renderer,
                            apply_first=all_cfg.relighting.apply_first,
                            use_conv=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        middle_block_ch = ch

        self.middle_block = time_embed_seq_module(
            resblock_normals_module(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=condition_dim,
                        condition_proj_dim=condition_proj_dim,
                        renderer=self.renderer,
                        apply_first=all_cfg.relighting.apply_first,
                        use_conv=True,
                    ),
            AttentionBlockNormals(
                ch,
                norm_type='GroupNorm',
                shading_channels=self.renderer.shading_channels,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            resblock_normals_module(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=condition_dim,
                        condition_proj_dim=condition_proj_dim,
                        renderer=self.renderer,
                        apply_first=all_cfg.relighting.apply_first,
                        use_conv=True,
                    ),
        )

        output_block_ch = []
        input_block_ch = input_block_chans[:]

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                '''
                Create resblock = num_res_blocks + 1
                '''
                ich = input_block_chans.pop()
                layers = [
                    resblock_normals_module(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=condition_dim,
                        condition_proj_dim=condition_proj_dim,
                        renderer=self.renderer,
                        apply_first=all_cfg.relighting.apply_first,
                        use_conv=True,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlockNormals(
                            ch,
                            norm_type='GroupNorm',
                            shading_channels=self.renderer.shading_channels,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
               
                if level and i == num_res_blocks:
                    '''
                    n = len(channel_mult)
                    In each level(n, n-1, ..., 2, 1, 0) except level=0, 
                    when i==num_res_block we create 
                    - upsample/downsample 
                    - resblock(if resblock_updown == True)
                    '''
                    out_ch = ch
                    layers.append(
                        resblock_normals_module(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            condition_dim=condition_dim,
                            condition_proj_dim=condition_proj_dim,
                            renderer=self.renderer,
                            apply_first=all_cfg.relighting.apply_first,
                            use_conv=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                
                output_block_ch.append(ch+ich)
                self.output_blocks.append(time_embed_seq_module(*layers))
                # print(i, layers)
                # print("*"*100)
        
        if all_cfg.relighting.mult_shaded == 'SharedConv':
            # Use same Conv layers to produce Img and shading
            self.out = nn.Sequential(
                normalization(channels=input_ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, out_channels+all_cfg.relighting.num_shaded_ch, 3, padding=1),
            )
        elif all_cfg.relighting.mult_shaded == 'SepConv':
            # Use individual Conv layers to produce Img and shading
            self.out = nn.Sequential(
                normalization(channels=input_ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, out_channels, 3, padding=1),
                conv_nd(dims, input_ch, all_cfg.relighting.num_shaded_ch, 3, padding=1),
            )
        else: 
            self.out = nn.Sequential(
                normalization(channels=input_ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
            )

        # print("#"*100)
        # print("Input blocks")
        # print("#"*100)
        # print(self.input_blocks)
        # print("#"*100)
        # exit()
        # print("Middle blocks")
        # print("#"*100)
        # print(self.middle_block)
        # print("#"*100)
        # print("Output blocks")
        # print("#"*100)
        # print(self.output_blocks)
        # print("#"*100)
        # exit()

        # print(input_block_ch)
        # print(middle_block_ch)
        # print(output_block_ch)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Input Blocks
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, condition=kwargs)
            hs.append(h)
        # Mid Block
        h = self.middle_block(h, emb, condition=kwargs)
        # Out Blocks
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, condition=kwargs)
        h = h.type(x.dtype)
        
        # Output Layer
        if self.all_cfg.relighting.mult_shaded == 'SepConv':
            out = self.out[:-2](h)
            img_branch = self.out[-1]
            shading_branch = self.out[-2]
            img = img_branch(out)
            shading_img = shading_branch(out)
            out = img * shading_img
        elif self.all_cfg.relighting.mult_shaded == 'SharedConv':
            out = self.out(h)
            img = out[:, :3, :, :]
            shading_img = out[:, 3:, :, :]
            out = img * shading_img
        else:
            out = self.out(h)


        return {'output':out}
