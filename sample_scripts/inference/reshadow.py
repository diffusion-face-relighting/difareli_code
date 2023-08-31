# from __future__ import print_function 
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--itp', nargs='+', default=None)
parser.add_argument('--itp_step', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--add_shadow', action='store_true', default=False)
parser.add_argument('--vary_shadow_range', nargs='+', type=float, default=None)
parser.add_argument('--vary_shadow', action='store_true', default=False)
# Samples selection
parser.add_argument('--idx', nargs='+', default=[])
parser.add_argument('--sample_pair_json', type=str, default=None)
parser.add_argument('--sample_pair_mode', type=str, default=None)
parser.add_argument('--src_dst', nargs='+', default=[], help='list of src and dst image')
# Rendering
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--scale_sh', type=float, default=1.0)
parser.add_argument('--add_sh', type=float, default=None)
parser.add_argument('--sh_grid_size', type=int, default=None)
parser.add_argument('--sh_span', type=float, default=None)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
# Diffusion
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--timestep_respacing', type=str, default="")
# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--eval_dir', type=str, default=None)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--save_vid', action='store_true', default=False)
parser.add_argument('--fps', action='store_true', default=False)
# Experiment
parser.add_argument('--fixed_render', action='store_true', default=False)
parser.add_argument('--fixed_shadow', action='store_true', default=False)

args = parser.parse_args()

import os, sys, glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL, cv2
import json
import copy
import time
import torchvision
import pytorch_lightning as pl
sys.path.insert(0, '../')
from guided_diffusion.script_util import (
    seed_all,
)
from guided_diffusion.tensor_util import (
    make_deepcopyable,
    dict_slice,
    dict_slice_se
)

from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca

# Sample utils
sys.path.insert(0, '../')
from sample_utils import (
    ckpt_utils, 
    params_utils, 
    vis_utils, 
    file_utils, 
    inference_utils, 
    mani_utils,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def make_condition(cond, src_idx, dst_idx, n_step=2, itp_func=None):
    condition_img = list(filter(None, dataset.condition_image))
    args.interpolate = args.itp
    misc = {'condition_img':condition_img,
            'src_idx':src_idx,
            'dst_idx':dst_idx,
            'n_step':n_step,
            'avg_dict':avg_dict,
            'dataset':dataset,
            'args':args,
            'itp_func':itp_func,
            'img_size':cfg.img_model.image_size,
            'deca_obj':deca_obj,
            'batch_size':args.batch_size,
            'cfg':cfg,
            }  
    cond['misc'] = misc
    
    if itp_func is not None:
        cond['use_render_itp'] = False 
    else:
        cond['use_render_itp'] = True
    # This is for the noise_dpm_cond_img
    if cfg.img_model.apply_dpm_cond_img:
        cond['image'] = th.stack([cond['image'][src_idx]] * n_step, dim=0)
        for k in cfg.img_model.dpm_cond_img:
            if 'faceseg' in k:
                cond[f'{k}_mask'] = th.stack([cond[f'{k}_mask'][src_idx]] * n_step, dim=0)
        
    cond, _ = inference_utils.build_condition_image(cond=cond, misc=misc)
    cond = inference_utils.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True)
    cond['cfg'] = cfg
    if (cfg.img_model.apply_dpm_cond_img) and (np.any(n is not None for n in cfg.img_model.noise_dpm_cond_img)):
        cond['use_cond_xt_fn'] = True
        for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
            cond[f'{k}_img'] = cond[f'{k}_img'].to(device)
            if p is not None:
                if 'dpm_noise_masking' in p:
                    cond[f'{k}_mask'] = cond[f'{k}_mask'].to(device)
                    cond['image'] = cond['image'].to(device)
    

    if 'render_face' in args.itp:
        interp_set = args.itp.copy()
        interp_set.remove('render_face')
    else:
        interp_set = args.itp
        
    # Interpolate non-spatial
    interp_cond = mani_utils.iter_interp_cond(cond, interp_set=interp_set, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func, add_shadow=args.add_shadow, vary_shadow=args.vary_shadow, vary_shadow_range=args.vary_shadow_range)
    cond.update(interp_cond)
        
    # Repeated non-spatial
    repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.itp + ['light', 'img_latent']))
    cond.update(repeated_cond)

    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    if cfg.img_cond_model.override_cond != '':
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    else:    
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector
    cond = inference_utils.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
    return cond
    
def ext_sub_step(n_step):
    sub_step = []
    bz = args.batch_size
    tmp = n_step
    while tmp > 0:
        if tmp - bz > 0:
            sub_step.append(bz)
        else:
            sub_step.append(tmp)
        tmp -= bz
    return np.cumsum([0] + sub_step)

def relight(dat, model_kwargs, itp_func, n_step=3, src_idx=0, dst_idx=1):
    '''
    Relighting the image
    Output : Tensor (B x C x H x W); range = -1 to 1
    '''
    # Rendering
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond, 
                        src_idx=src_idx, dst_idx=dst_idx, 
                        n_step=n_step, itp_func=itp_func
                    )

    # Reverse 
    cond_rev = copy.deepcopy(cond)
    cond_rev = dict_slice(in_d=cond_rev, keys=cond_rev.keys(), n=1) # Slice only 1st image out for inversion
    if cfg.img_cond_model.apply:
        cond_rev = pl_sampling.forward_cond_network(model_kwargs=cond_rev)
        
    print("[#] Apply Mean-matching...")
    reverse_ddim_sample = pl_sampling.reverse_proc(x=dat[0:1, ...], model_kwargs=cond_rev, store_mean=True)
    noise_map = reverse_ddim_sample['final_output']['sample']
    rev_mean = reverse_ddim_sample['intermediate']
    
    #NOTE: rev_mean WILL BE MODIFIED; This is for computing the ratio of inversion (brightness correction).
    sample_ddim = pl_sampling.forward_proc(
        noise=noise_map,
        model_kwargs=cond_rev,
        store_intermediate=False,
        rev_mean=rev_mean)

    assert noise_map.shape[0] == 1
    rev_mean_first = [x[:1] for x in rev_mean]
    
    print("[#] Relighting...")
    sub_step = ext_sub_step(n_step)
    relit_out = []
    for i in range(len(sub_step)-1):
        print(f"[#] Sub step relight : {sub_step[i]} to {sub_step[i+1]}")
        start = sub_step[i]
        end = sub_step[i+1]
        
        # Relight!
        mean_match_ratio = copy.deepcopy(rev_mean_first)
        cond['use_render_itp'] = True
        cond_relight = copy.deepcopy(cond)
        cond_relit = dict_slice_se(in_d=cond_relight, keys=cond_relight.keys(), s=start, e=end) # Slice only 1st image out for inversion
        if cfg.img_cond_model.apply:
            cond_relit = pl_sampling.forward_cond_network(model_kwargs=cond_relit)
        
        relight_out = pl_sampling.forward_proc(
            noise=th.repeat_interleave(noise_map, repeats=end-start, dim=0),
            model_kwargs=cond_relit,
            store_intermediate=False,
            add_mean=mean_match_ratio)
        
        relit_out.append(relight_out["final_output"]["sample"].detach().cpu().numpy())
    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))
    
    return relit_out, cond['cond_img']

if __name__ == '__main__':
    seed_all(args.seed)
    if args.postfix != '':
        args.postfix = f'_{args.postfix}'
    # Load Ckpt
    if args.cfg_name is None:
        args.cfg_name = args.log_dir + '.yaml'
    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg
    
    print(f"[#] Sampling with diffusion_steps = {args.diffusion_steps}")
    print(f"[#] Sampling with timestep respacing = {args.timestep_respacing}")
    cfg.diffusion.diffusion_steps = args.diffusion_steps
    cfg.diffusion.timestep_respacing = args.timestep_respacing
    model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)
    model_dict = inference_utils.eval_mode(model_dict)

    # Load dataset
    if args.dataset == 'itw':
        img_dataset_path = f"/data/mint/DPM_Dataset/ITW/itw_images_aligned/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ITW/params/"
        img_ext = '.png'
        cfg.dataset.training_data = 'ITW'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/itw_images_aligned/'
    elif args.dataset == 'ffhq':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = 'ffhq_256_with_anno'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
    elif args.dataset in ['mp_valid', 'mp_test', 'mp_test2']:
        if args.dataset == 'mp_test':
            sub_f = '/MultiPIE_testset/'
        elif args.dataset == 'mp_test2':
            sub_f = '/MultiPIE_testset2/'
        elif args.dataset == 'mp_valid':
            sub_f = '/MultiPIE_validset/'
        else: raise ValueError
        img_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/mp_aligned/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/params/"
        img_ext = '.png'
        cfg.dataset.training_data = f'/MultiPIE/{sub_f}/'
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/mp_aligned/'
    else: raise ValueError

    cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
    cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
    cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
    cfg.dataset.laplacian_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/eyes_segment/"
    cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"

    loader, dataset, avg_dict = load_data_img_deca(
        data_dir=img_dataset_path,
        deca_dir=deca_dataset_path,
        batch_size=int(1e5),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
        set_=args.set,
        cfg=cfg,
        img_ext=img_ext,
        mode='sampling'
    )
    
    data_size = dataset.__len__()
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
    all_img_idx, all_img_name, n_subject = mani_utils.get_samples_list(args.sample_pair_json, 
                                                                            args.sample_pair_mode, 
                                                                            args.src_dst, img_path, 
                                                                            -1)
    #NOTE: Initialize a DECA renderer
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]):
        mask = params_utils.load_flame_mask()
    else: mask=None
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Run from start->end idx
    start, end = int(args.idx[0]), int(args.idx[1])
    if end > n_subject:
        end = n_subject 
    if start >= n_subject: raise ValueError("[#] Start beyond the sample index")
    print(f"[#] Run from index of {start} to {end}...")
        
    for i in range(start, end):
        img_idx = all_img_idx[i]
        img_name = all_img_name[i]
        
        dat = th.utils.data.Subset(dataset, indices=img_idx)
        subset_loader = th.utils.data.DataLoader(dat, batch_size=2,
                                            shuffle=False, num_workers=24)
                                   
        dat, model_kwargs = next(iter(subset_loader))
        print("#"*100)
        # Indexing
        src_idx = 0
        dst_idx = 1
        src_id = img_name[0]
        dst_id = img_name[1]
        # LOOPER SAMPLING
        n_step = args.itp_step
        print(f"[#] Current idx = {i}, Set = {args.set}, Src-id = {src_id}, Dst-id = {dst_id}")
        
        pl_sampling = inference_utils.PLSampling(model_dict=model_dict,
                                                    diffusion=diffusion,
                                                    reverse_fn=diffusion.ddim_reverse_sample_loop,
                                                    forward_fn=diffusion.ddim_sample_loop,
                                                    denoised_fn=None,
                                                    cfg=cfg,
                                                    args=args)
        
        model_kwargs = inference_utils.prepare_cond_sampling(cond=model_kwargs, cfg=cfg)
        model_kwargs['sample_idx'] = i
        model_kwargs['cfg'] = cfg
        model_kwargs['use_cond_xt_fn'] = False
        if (cfg.img_model.apply_dpm_cond_img) and (np.any(n is not None for n in cfg.img_model.noise_dpm_cond_img)):
            model_kwargs['use_cond_xt_fn'] = True
            for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
                model_kwargs[f'{k}_img'] = model_kwargs[f'{k}_img'].to(device)
                if p is not None:
                    if 'dpm_noise_masking' in p:
                        model_kwargs[f'{k}_mask'] = model_kwargs[f'{k}_mask'].to(device)
                        model_kwargs['image'] = model_kwargs['image'].to(device)
           
        itp_fn = mani_utils.slerp if args.slerp else mani_utils.lerp
        itp_fn_str = 'Slerp' if itp_fn == mani_utils.slerp else 'Lerp'
        itp_str = '_'.join(args.itp)
        
        model_kwargs['use_render_itp'] = True
        out_relit, out_render = relight(dat = dat,
                                    model_kwargs=model_kwargs,
                                    src_idx=src_idx, dst_idx=dst_idx,
                                    itp_func=itp_fn,
                                    n_step = n_step
                                )
        
        #NOTE: Save result
        out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{itp_str}/reverse_sampling/"
        os.makedirs(out_dir_relit, exist_ok=True)
        save_res_dir = f"{out_dir_relit}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/n_frames={n_step}/"
        os.makedirs(save_res_dir, exist_ok=True)
        
        f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
        vis_utils.save_images(path=f"{save_res_dir}", fn="res", frames=f_relit)
        
        if args.eval_dir is not None:
            # if args.dataset in ['mp_valid', 'mp_test']
            # eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/out/{args.dataset}/"
            eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/out/"
            os.makedirs(eval_dir, exist_ok=True)
            torchvision.utils.save_image(tensor=f_relit[-1], fp=f"{eval_dir}/input={src_id}#pred={dst_id}.png")
            
        
        is_render = True if out_render is not None else False
        if is_render:
            clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
            if clip_ren:
                vis_utils.save_images(path=f"{save_res_dir}", fn="ren", frames=(out_render + 1) * 0.5)
            else:
                vis_utils.save_images(path=f"{save_res_dir}", fn="ren", frames=out_render[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0)
                
        if args.save_vid:
            """
            save the video
            Args:
                frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
                fn : path + filename to save
                fps : video fps
            """
            #NOTE: save_video, w/ shape = TxHxWxC and value range = [0, 255]
            vid_relit = out_relit
            vid_relit = vid_relit.permute(0, 2, 3, 1)
            vid_relit = ((vid_relit + 1)*127.5).clamp_(0, 255).type(th.ByteTensor)
            vid_relit_rt = th.cat((vid_relit, th.flip(vid_relit, dims=[0])))
            torchvision.io.write_video(video_array=vid_relit, filename=f"{save_res_dir}/res.mp4", fps=args.fps)
            torchvision.io.write_video(video_array=vid_relit_rt, filename=f"{save_res_dir}/res_rt.mp4", fps=args.fps)
            if is_render:
                out_render = out_render[:, :3]
                vid_render = out_render
                # vid_render = th.cat((out_render, th.flip(out_render, dims=[0])))
                clip_ren = False #if 'wclip' in dataset.condition_image else True
                if clip_ren:
                    vid_render = ((vid_render.permute(0, 2, 3, 1) + 1) * 127.5).clamp_(0, 255).type(th.ByteTensor)
                    torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
                else:
                    vid_render = (vid_render.permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255)).type(th.ByteTensor)
                    torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
                    vid_render_rt = th.cat((vid_render, th.flip(vid_render, dims=[0])))
                    torchvision.io.write_video(video_array=vid_render_rt, filename=f"{save_res_dir}/ren_rt.mp4", fps=args.fps)
                
        with open(f'{save_res_dir}/res_desc.json', 'w') as fj:
            log_dict = {'sampling_args' : vars(args), 
                        'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_fn':itp_fn_str, 'itp':itp_str}}
            json.dump(log_dict, fj)
            
            
    # Free memory!!!
    del deca_obj               
