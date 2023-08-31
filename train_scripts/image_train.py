"""
Train a diffusion model on images.
"""

import os
import pytorch_lightning as pl
from guided_diffusion import logger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from config.base_config import parse_args
from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_img_and_diffusion,
    seed_all,
)
from guided_diffusion.train_util.cond_train_util import TrainLoop

def main():
    cfg = parse_args()
    seed_all(47)    # Seeding the model - Independent training

    logger.configure(dir=cfg.train.log_dir)
    logger.log("[#] Creating model and diffusion...")

    img_model, diffusion = create_img_and_diffusion(cfg)
    print(img_model)
    # Filtered out the None model
    img_model = {k: v for k, v in img_model.items() if v is not None}
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    logger.log("[#] Creating data loader...")
    train_loader, _, _ = load_data_img_deca(
        data_dir=cfg.dataset.data_dir,
        deca_dir=cfg.dataset.deca_dir,
        batch_size=cfg.train.batch_size,
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
        cfg=cfg,
    )

    logger.log("[#] Training...")
    
    print(f"Initialize \"{cfg.train.logger_mode}\" logger : {cfg.train.logger_dir}")
    os.makedirs(cfg.train.logger_dir, exist_ok=True)
    if cfg.train.logger_mode == 'wandb':
        t_logger = WandbLogger(project='Relighting-DPM', save_dir=cfg.train.logger_dir, tags=[cfg.train_misc.exp_name], name=cfg.train_misc.cfg_name)
    elif cfg.train.logger_mode == 'tb':
        t_logger = TensorBoardLogger(save_dir=cfg.train.logger_dir, name="diffusion", version=cfg.train_misc.exp_name, sub_dir=cfg.train_misc.cfg_name)
    else: 
        raise NotImplementedError("[#] Logger mode is not available")

    train_loop = TrainLoop(
        model=list(img_model.values()),
        name=list(img_model.keys()),
        diffusion=diffusion,
        train_loader=train_loader,
        cfg=cfg,
        t_logger=t_logger,
        schedule_sampler=schedule_sampler,
    )
    
    train_loop.run()

if __name__ == "__main__":
    main()
