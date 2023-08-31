import argparse


from . import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.models.unet import EncoderUNetModelNoTime, UNetModelCondition, UNetModel
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_concat import EncoderUNet_SpatialCondition, UNetModel_SpatialCondition_Concat
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart import UNetModel_SpatialCondition_Hadamart
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_no_dpm import UNetModel_SpatialCondition_Hadamart_No_DPM
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_no_dpm_notime import UNetModel_SpatialCondition_Hadamart_No_DPM_NoTime
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_sameemb import UNetModel_SpatialCondition_Hadamart_SameEmb
from guided_diffusion.models.unet_duplicate import UNetModelConditionDuplicate
from guided_diffusion.models.normals_arch.unet_N_Last import UNetNormalsLastLayer
from guided_diffusion.models.normals_arch.unet_N_All import UNetNormalsAll
from guided_diffusion.models.dense import DenseDDPM, AutoEncoderDPM, DenseDDPMCond

NUM_CLASSES = 1000

# Pipeline
def create_img_deca_and_diffusion(cfg):
    img_model = create_model(cfg.img_model)
    params_model = create_param_model(cfg.param_model)
    diffusion = create_gaussian_diffusion(cfg.diffusion)
    return img_model, params_model, diffusion

def create_deca_and_diffusion(cfg):
    if cfg.param_model.conditioning:
        param_model = create_param_model(cfg=cfg.param_model, cfg_cond=cfg)
    else:
        param_model = create_param_model(cfg=cfg.param_model)
    diffusion = create_gaussian_diffusion(cfg.diffusion)
    return param_model, diffusion

def create_img_and_diffusion(cfg):
    img_model = create_model(cfg.img_model, all_cfg=cfg)
    if cfg.img_cond_model.apply:
        img_cond_model = create_model(cfg.img_cond_model, all_cfg=cfg)
    else: img_cond_model = None
    diffusion = create_gaussian_diffusion(cfg.diffusion)
    
    return {cfg.img_model.name:img_model, cfg.img_cond_model.name:img_cond_model}, diffusion

# Each sub-modules
def create_param_model(cfg, cfg_cond=None):
    if cfg.deca_cond:
        img_model, _ = create_img_and_diffusion(cfg_cond)
        if cfg.arch == 'magenta':
            return DenseDDPMCond(
                in_channels=cfg.in_channels,
                model_channels=cfg.model_channels,
                num_layers=cfg.num_layers,
                use_checkpoint=cfg.use_checkpoint,
                encoder=img_model)
        else: raise NotImplementedError
    else:
        if cfg.arch == 'magenta':
            return DenseDDPM(
                in_channels=cfg.in_channels,
                model_channels=cfg.model_channels,
                num_layers=cfg.num_layers,
                use_checkpoint=cfg.use_checkpoint,
            )
        elif cfg.arch == 'autoenc':
            return AutoEncoderDPM(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                model_channels=cfg.model_channels,
                num_layers=cfg.num_layers,
                use_checkpoint=cfg.use_checkpoint,
            )
        else: raise NotImplementedError

def create_model(cfg, all_cfg=None):
    if cfg.channel_mult == "":
        if cfg.image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif cfg.image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfg.image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif cfg.image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {cfg.image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in cfg.channel_mult.split(","))

    attention_ds = []
    for res in cfg.attention_resolutions.split(","):
        attention_ds.append(cfg.image_size // int(res))
    if cfg.arch == 'UNet':
        return UNetModel(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
        )
    elif cfg.arch == 'UNetNormalsLastLayer':
        return UNetNormalsLastLayer(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'UNetNormalsAll':
        return UNetNormalsAll(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'UNetCond':
        return UNetModelCondition(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=True,
        )
    elif cfg.arch == 'EncoderUNet':
        return EncoderUNetModelNoTime(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            conditioning=True,
            pool=cfg.pool
        )
    elif cfg.arch == 'UNetCondDuplicate':
            return UNetModelConditionDuplicate(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=True,
            num_SH=all_cfg.relighting.num_SH,
            last_conv=cfg.last_conv
        )
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart':
        return UNetModel_SpatialCondition_Hadamart(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart_No_DPM':
        return UNetModel_SpatialCondition_Hadamart_No_DPM(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
        
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart_No_DPM_NoTime':
        return UNetModel_SpatialCondition_Hadamart_No_DPM_NoTime(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart_SameEmb':
        return UNetModel_SpatialCondition_Hadamart_SameEmb(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'UNetCond_SpatialCondition_Concat':
        return UNetModel_SpatialCondition_Concat(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=True,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'EncoderUNet_SpatialCondition':
        return EncoderUNet_SpatialCondition(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            conditioning=True,
            pool=cfg.pool
        )
    else: raise NotImplementedError

def create_gaussian_diffusion(cfg):
    betas = gd.get_named_beta_schedule(cfg.noise_schedule, cfg.diffusion_steps)
    if cfg.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif cfg.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not cfg.timestep_respacing:
        timestep_respacing = [cfg.diffusion_steps]
    return SpacedDiffusion(
        # use_timesteps=space_timesteps(cfg.diffusion_steps, timestep_respacing),
        use_timesteps=space_timesteps(cfg.diffusion_steps, cfg.timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not cfg.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cfg.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=cfg.rescale_timesteps,
    )

# Utils
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def seed_all(seed: int):

    """
    Seeding everything for paired indendent training

    :param seed: seed number for a number generator.
    """

    import os
    import numpy as np
    import torch as th
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
def compare_models(model_1, model_2):
    import torch as th
    models_differ = 0
    counter = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if th.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
        counter+=1
    if models_differ == 0:
        print('Models match perfectly! :)')
    else: print(f'Mismatch {models_differ}/{counter} layers')
    
    
def dump_model_params(model, fn):
    txt = ""
    if '.txt' not in fn:
        fn += '.txt'
    for k, v in model.named_parameters():
        txt += f"{k}, {v}\n"
    with open(fn, 'w') as f:
        f.write(txt)
    f.close()