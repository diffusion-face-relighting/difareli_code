import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, params_utils
import cv2, PIL
import time

class PLSampling(pl.LightningModule):
    def __init__(self, 
                 model_dict, 
                 diffusion, 
                 forward_fn, 
                 reverse_fn, 
                 cfg, 
                 args=None, 
                 denoised_fn=None):
        
        super(PLSampling, self).__init__()
        self.forward_fn = forward_fn
        self.reverse_fn = reverse_fn
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg
        self.args = args
        self.const_noise = th.randn((len(cfg.img_model.dpm_cond_img), 3, cfg.img_model.image_size, cfg.img_model.image_size)).cuda()
        
    def forward_cond_network(self, model_kwargs):
        with th.no_grad():
            if self.cfg.img_cond_model.apply:
                x = model_kwargs['cond_img'].cuda().float()
                img_cond = self.model_dict[self.cfg.img_cond_model.name](
                    x=x,
                    emb=None,
                )
                # Override the condition
                if self.cfg.img_cond_model.override_cond != "":
                    model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
                else: raise NotImplementedError
        return model_kwargs

    def reverse_proc(self, x, model_kwargs, progress=True, store_intermediate=True, store_mean=False):
        # Mimic the ddim_sample_loop or p_sample_loop
        model_kwargs['const_noise'] = self.const_noise
        if self.reverse_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.reverse_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                denoised_fn = self.denoised_fn,
                model_kwargs=model_kwargs,
                progress=progress,
                store_intermidiate=store_intermediate,
                cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
                store_mean=store_mean
            )
        else: raise NotImplementedError

        # if store_intermediate:
            # assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self,
                     model_kwargs,
                     noise,
                     store_intermediate=True,
                     sdedit=None,
                     rev_mean=None,
                     add_mean=None):
        model_kwargs['const_noise'] = self.const_noise
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs,
            store_intermidiate=store_intermediate,
            cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
            sdedit=sdedit,
            rev_mean=rev_mean,
            add_mean=add_mean
        )
        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def override_modulator(self, cond, val=1):
        print(f"[#] Override the modulator with {val}")
        for i in range(len(cond)):
            if val == 1:
                cond[i] = th.ones_like(cond[i])
        return cond
    
def cond_xt_fn(cond, cfg, use_render_itp, t, diffusion, noise, device='cuda'):
    #NOTE: This specifically run for ['dpm_cond_img']
    
    def dpm_noise(x_start, p, k, noise, i):
        if p is None: return x_start
        share = True if p.split('-')[0] == 'share' else False
        masking = p.split('-')[-1]
        
        #NOTE: Repeat the noise to used following share/sep noise
        if share:
            noise = th.repeat_interleave(input=noise[0:1], dim=0, repeats=x_start.shape[0])
        else:
            noise = th.repeat_interleave(input=noise[[i]], dim=0, repeats=x_start.shape[0])
        if masking == 'dpm_noise_masking':
            img = cond['image']
            mask =  cond[f'{k}_mask'].bool()
            assert th.all(mask == cond[f'{k}_mask'])
            xt = (diffusion.q_sample(img, t, noise=noise) * mask) + (-th.ones_like(img) * ~mask)
        elif masking == 'dpm_noise':
                xt = diffusion.q_sample(x_start, t, noise=noise)
        else: raise NotImplementedError("[#] Only dpm_noise_masking and dpm_noise is available")
        return xt
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for i, (k, p) in enumerate(zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img)):
            if ('faceseg' in k) or ('deca' in k): # if 'faceseg' in k:
                if use_render_itp:
                    xt = dpm_noise(x_start=cond[f'{k}'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
                else:
                    xt = dpm_noise(x_start=cond[f'{k}_img'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
            else:
                if use_render_itp: 
                    tmp_img = cond[f'{k}']
                else: 
                    tmp_img = cond[f'{k}_img']
                dpm_cond_img.append(tmp_img)
        cond['dpm_cond_img'] = th.cat(dpm_cond_img, dim=1)
    else:
        cond['dpm_cond_img'] = None
    return cond

def get_init_noise(n, mode, img_size, device):
    '''
    Return the init_noise used as input.
    :params mode: mode for sampling noise => 'vary_noise', 'fixed_noise'
    '''
    if mode == 'vary_noise':
        init_noise = th.randn((n, 3, img_size, img_size))
    elif mode == 'fixed_noise':
        init_noise = th.cat([th.randn((1, 3, img_size, img_size))] * n, dim=0)
    else: raise NotImplementedError

    return init_noise.to(device)

def to_tensor(cond, key, device):
    for k in key:
        if k not in cond.keys():
            continue
        else:
            if isinstance(cond[k], list):
                for i in range(len(cond[k])):
                    cond[k][i] = th.tensor(cond[k][i]).to(device)
            else:
                if th.is_tensor(cond[k]):
                    cond[k] = cond[k].clone().to(device)
                elif isinstance(cond[k], np.ndarray):
                    cond[k] = th.tensor(cond[k]).to(device)
                else:
                    continue
    return cond

def eval_mode(model_dict):
    for k, _ in model_dict.items():
        model_dict[k].eval()
    return model_dict

def prepare_cond_sampling(cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    ###Note: 
     - cond[f'{k}'_img] is the original one from dataloader
     - cond[f'{k}'] is the original one render & build_condition_image() fn
    """
    
    print(cond.keys())
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k in cfg.img_model.dpm_cond_img:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            dpm_cond_img.append(tmp_img.to(device))
        cond['dpm_cond_img'] = th.cat((dpm_cond_img), dim=1)
    else:
        cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        cond_img = []
        for k in cfg.img_cond_model.in_image:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            print(k, tmp_img.shape)
            cond_img.append(tmp_img.to(device))
        cond['cond_img'] = th.cat((cond_img), dim=1).to(device)
    else:
        cond['cond_img'] = None
        
    return cond

def build_condition_image(cond, misc):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]) or np.any(['shadow_mask' in i for i in condition_img]):
        # Render the face
        if args.rotate_normals:
            #NOTE: Render w/ Rotated normals; cond['light'] shape = B x 27
            cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light']))
            cond['R_normals'] = params_utils.get_R_normals(n_step=n_step)
            lb = cond['light'].copy()
            lr_shading = int(n_step//2)
            # cond['light'][3:lr_shading] *= args.scale_sh
            cond['light'] *= args.scale_sh
            if args.add_sh is not None:
                cond['light'][3:lr_shading+3, 0:3] += args.add_sh
            if args.diffuse_sh is not None:
                cond['light'][3:lr_shading+3, 0:3] += args.diffuse_sh
                cond['light'][3:lr_shading+3, 3:] -= args.diffuse_sh
            if args.diffuse_perc is not None:
                for i, sh_idx in enumerate(np.arange(0, 27, 3)):
                    # print(sh_idx)
                    if i == 0:
                        cond['light'][3:lr_shading+3, sh_idx:sh_idx+3] -= (np.mean(cond['light'][3:lr_shading+3, sh_idx:sh_idx+3], axis=1, keepdims=True) * args.diffuse_perc)
                    else:
                        cond['light'][3:lr_shading+3, sh_idx:sh_idx+3] += (np.mean(cond['light'][3:lr_shading+3, sh_idx:sh_idx+3], axis=1, keepdims=True) * args.diffuse_perc)
            print(f"[#] Mean light after scale with {args.scale_sh}: {np.mean(lb)} -> {np.mean(cond['light'])}")
        elif args.sh_grid_size is not None:
            #NOTE: Render w/ grid light 
            cond['light'] = params_utils.grid_sh(sh=cond['light'][src_idx], n_grid=args.sh_grid_size, sx=args.sh_span_x, sy=args.sh_span_y, sh_scale=args.sh_scale, use_sh=args.use_sh).reshape(-1, 27)
        elif 'render_face' in args.interpolate:
            #NOTE: Render w/ interpolated light (Mainly use this)
            interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        # elif 'render_face' in args.interpolate:
            #NOTE: Render w/ interpolated light (For aj jumras or any itw)
            # print(cond['light'])
            # print(cond['mod_light'])
            # interp_cond = mani_utils.iter_interp_cond_lightfile(cond, itp_src='light', itp_dst='mod_light', src_idx=src_idx, n_step=n_step, interp_fn=itp_func)
            # print(interp_cond)
            # cond.update(interp_cond)
        elif 'render_face_modSH' in args.interpolate:
            #NOTE: Render w/ interpolated light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            # mod_SH = np.array([1, 1.1, 1.2, 1/1.1, 1/1.2])[..., None]
            mod_SH = np.array([0, 1, 1.2, 1/1.2])[..., None]
            repeated_cond['light'] = repeated_cond['light'] * mod_SH
            cond.update(repeated_cond)
        else:
            #NOTE: Render w/ same light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            cond.update(repeated_cond)
            # interp_cond = mani_utils.iter_interp_cond(cond, interp_set=args.interpolate, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            # interp_cond['pose'] = th.tensor(interp_cond['pose'])
            # cond.update(interp_cond)
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]) or np.any(['shadow_mask' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #TODO: Render DECA in minibatch
        sub_step = mani_utils.ext_sub_step(n_step, batch_size)
        all_render = []
        all_shadow_mask = []
        for i in range(len(sub_step)-1):
            print(f"[#] Sub step rendering : {sub_step[i]} to {sub_step[i+1]}")
            start = sub_step[i]
            end = sub_step[i+1]
            sub_cond = cond.copy()
            sub_cond['light'] = sub_cond['light'][start:end, :]
            # Deca rendered : B x 3 x H x W
            deca_rendered, orig_visdict = params_utils.render_deca(deca_params=sub_cond, 
                                                                idx=src_idx, n=end-start, 
                                                                avg_dict=avg_dict, 
                                                                render_mode=args.render_mode, 
                                                                rotate_normals=args.rotate_normals, 
                                                                mask=mask,
                                                                deca_obj=deca_obj,
                                                                repeat=True)
            # Shadow_mask : B x H x W
            # shadow_mask = params_utils.render_shadow_mask(
            #                                 sh_light=sub_cond['light'], 
            #                                 cam=sub_cond['cam'][src_idx],
            #                                 verts=orig_visdict['trans_verts_orig'], 
            #                                 deca=deca_obj)
            all_render.append(deca_rendered)
            # all_shadow_mask.append(shadow_mask[:, None, ...])
        print("Rendering time : ", time.time() - start_t)
        
        if args.fixed_render:
            print("[#] Fixed the Deca renderer")
            deca_rendered = all_render[0].repeat_interleave(repeats=len(all_render), dim=0)
        else:
            deca_rendered = th.cat(all_render, dim=0)
        # if args.fixed_shadow:
        #     print("[#] Fixed the Shadow mask")
        #     shadow_mask = all_shadow_mask[0].repeat_interleave(repeats=len(all_render), dim=0)
        # else:
        #     shadow_mask = th.cat(all_shadow_mask, dim=0)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            # mask_bg = cond['faceseg_nohead_mask'][[src_idx]]
            # sd = 58.050383371049826/127.5
            # val = 1
            # rpl_bg = th.normal(mean=val, std=sd, size=bg_tmp.shape)
            # bg_tmp = mask_bg * rpl_bg
            # import torchvision
            # torchvision.utils.save_image(cond['faceseg_nohead_mask'].double(), 'ggex.png')
            # torchvision.utils.save_image(bg_tmp, 'ggex2.png')
            # exit()
                                              
            # print(condition_img)
            # print(cond['faceseg_nohead_mask'].shape)
            # import torchvision
            # torchvision.utils.save_image(cond['faceseg_nohead_mask'].double(), 'ggex.png')
            # print(cond.keys())
            # exit()
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(n_step):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
        elif 'shadow_mask' in cond_img_name:
            shadow_mask_tmp = []
            for j in range(n_step):
                sm_tmp = shadow_mask[j].mul(255).add_(0.5).clamp_(0, 255)
                sm_tmp = sm_tmp.repeat_interleave(repeats=3, dim=0)
                sm_tmp = np.transpose(sm_tmp.cpu().numpy(), (1, 2, 0))  # HxWxC
                sm_tmp = sm_tmp.astype(np.uint8)
                sm_tmp = dataset.augmentation(PIL.Image.fromarray(sm_tmp))
                sm_tmp = dataset.prep_cond_img(sm_tmp, cond_img_name, i)
                sm_tmp = np.transpose(sm_tmp, (2, 0, 1))    # CxHxW
                sm_tmp = (sm_tmp / 127.5) - 1
                shadow_mask_tmp.append(sm_tmp[[0], ...])
            shadow_mask_tmp = np.stack(shadow_mask_tmp, axis=0)
            cond[cond_img_name] = th.tensor(shadow_mask_tmp).cuda()
            # print(cond['shadow_mask'].shape)
            # print(th.min(cond['shadow_mask']))
            # print(th.max(cond['shadow_mask']))
            # import torchvision
            # torchvision.utils.save_image((cond['shadow_mask']+1)*127.5/255.0, 'ggez.png')
            # exit()
    return cond, clip_ren


def build_condition_image_for_vids(cond, misc):
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]):
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #TODO: Render DECA in minibatch
        all_render = []
        print(f"[#] Rendering...")
        # Deca rendered : B x 3 x H x W
        deca_rendered, orig_visdict = params_utils.render_deca_videos(
                                                            deca_params=cond, 
                                                            avg_dict=avg_dict, 
                                                            render_mode=args.render_mode, 
                                                            rotate_normals=args.rotate_normals, 
                                                            mask=mask,
                                                            deca_obj=deca_obj)
        all_render.append(deca_rendered)
        print("Rendering time : ", time.time() - start_t)
        deca_rendered = th.cat(all_render, dim=0)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name):
            bg_tmp = cond[f"{cond_img_name}_img"]
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(deca_rendered.shape[0]):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
    return cond, clip_ren

def build_condition_image_rotateSH(cond, misc):
    src_idx = misc['src_idx']
    sub_step = misc['sub_step']
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]):
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        all_render = []
        print(f"[#] Rendering ({sub_step} steps)...")
        # Deca rendered : B x 3 x H x W
        deca_rendered, orig_visdict = params_utils.render_deca_rotateSH(
                                                            deca_params=cond, 
                                                            avg_dict=avg_dict, 
                                                            render_mode=args.render_mode, 
                                                            rotate_normals=args.rotate_normals, 
                                                            mask=mask,
                                                            deca_obj=deca_obj)
        all_render.append(deca_rendered)
        print("Rendering time : ", time.time() - start_t)
        deca_rendered = th.cat(all_render, dim=0)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * sub_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
            
            # bg_tmp = cond[f"{cond_img_name}_img"]
            # cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
            # # faceseg_use = f"{cond_img_name}_img" if f"{cond_img_name}_img" in list(cond.keys()) else f"{cond_img_name}"
            
            # # bg_tmp = [cond[f"{faceseg_use}"][src_idx]] * sub_step
            # # if th.is_tensor(cond[f"{faceseg_use}"][src_idx]):
            # #     bg_tmp = th.stack(bg_tmp, axis=0)
            # # else:
            # #     bg_tmp = np.stack(bg_tmp, axis=0)
            # bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * sub_step
            # if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
            #     bg_tmp = th.stack(bg_tmp, axis=0)
            # else:
            #     bg_tmp = np.stack(bg_tmp, axis=0)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(deca_rendered.shape[0]):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
    return cond, clip_ren
