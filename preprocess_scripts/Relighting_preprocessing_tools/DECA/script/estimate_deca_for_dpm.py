# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import argparse
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch as th
import os

sys.path.insert(0, '../')
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
sys.path.insert(0, '../../mothership_v16/')
from sample_scripts.sample_utils.params_utils import get_params_set


parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                    help='path to the test data, can be image folder, image path, image list, video')
parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                    help='path to the output directory, where results(obj, txt files) will be stored.')

parser.add_argument('--save_images_folder', default='TestSamples/examples/results', type=str,
                    help='path to the output directory, where results(obj, txt files) will be stored.')
parser.add_argument('--save_params_folder', default='TestSamples/examples/params', type=str,
                    help='path to the output directory, where results(obj, txt files) will be stored.')
parser.add_argument('--device', default='cuda', type=str,
                    help='set device, cpu for using cpu' )
# process test images
parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to crop input image, set false only when the test image are well cropped' )
parser.add_argument('--sample_step', default=10, type=int,
                    help='sample images from video data for every step' )
parser.add_argument('--detector', default='fan', type=str,
                    help='detector for cropping face, check decalib/detectors.py for details' )
# rendering option
parser.add_argument('--rasterizer_type', default='standard', type=str,
                    help='rasterizer type: pytorch3d or standard' )
parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to render results in original image size, currently only works when rasterizer_type=standard')
# save
parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to use FLAME texture model to generate uv texture map, \
                        set it to True only if you downloaded texture model' )
parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization of output' )
parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save 2D and 3D keypoints' )
parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save depth image' )
parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                        Note that saving objs could be slow' )
parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save outputs as .mat')
parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization output as seperate images' )
parser.add_argument('--useTemplate', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization output as seperate images' )
parser.add_argument('--useAvgCam', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization output as seperate images' )
parser.add_argument('--useAvgTform', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization output as seperate images' )
parser.add_argument('--set', default='examples', type=str,
                    help='Specified the [train/valid/test] of dataset' )
parser.add_argument('--params_prefix', default='examples', type=str,
                    help='Prefix saving the params' )
parser.add_argument('--index', default=-1, type=int, nargs="+",
                    help='Prefix saving the params' )
parser.add_argument('--fast_save_params', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='Bypass rendering pipeline, for save predicted params only' )
parser.add_argument('--masking_flame', default=False, action='store_true')

args = parser.parse_args()


def mean_face():
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'tform']
    deca_params_train = get_params_set('train', params_key = params_key, path="/data/mint/ffhq_256_with_anno/params_finale/")
    shape = []; pose = []; exp = []; cam = []; light = []; tform = [];
    for k, v in deca_params_train.items():
        shape.append(th.tensor(v['shape']))
        pose.append(th.tensor(v['pose']))
        exp.append(th.tensor(v['exp']))
        cam.append(th.tensor(v['cam']))
        light.append(th.tensor(v['light']))
        tform.append(th.tensor(v['tform']))
    shape = th.stack(shape)
    pose = th.stack(pose)
    exp = th.stack(exp)
    cam = th.stack(cam)
    light = th.stack(light)
    tform = th.stack(tform)
    return {'shape':shape, 'pose':pose, 'exp':exp, 'cam':cam, 'light':light, 'tform':tform}

def main():
    # print("ARGS : ", args)
    device = args.device
    
    if args.useAvgCam:
        mean_cam = th.mean(mean_face_dict['cam'], dim=0, keepdims=True).float().to(device)
    else: mean_cam = None
    if args.useAvgTform:
        mean_tform = th.mean(mean_face_dict['tform'], dim=0).float().to(device)
        mean_tform = (mean_tform.reshape(3, 3))[None, ...]

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    mask_dir = os.path.join(f'{os.path.abspath(os.path.dirname(__file__))}/../data/')
    if args.masking_flame:
        f_mask = np.load(f'{mask_dir}/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
        v_mask = np.load(f'{mask_dir}/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
        mask={
            'v_mask':v_mask['face'].tolist(),
            'f_mask':f_mask['face'].tolist()
        }
    else: mask=None

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device, mode='deca', mask=mask)
    
    if args.index == -1:
        data_iters = range(len(testdata))
        print(f"[#] Process all data : Total={len(data_iters)}")
    else:
        start, end = args.index[0], args.index[1]
        assert start < end
        assert end <= len(testdata)
        assert start <= len(testdata)
        data_iters = range(start, end)
        print(f"[#] Process at {start}->{end} : Total={len(data_iters)}")
        
    #NOTE: SAVING PARAMS
    if args.save_params_folder is not None:
        os.makedirs(args.save_params_folder, exist_ok=True)
        if args.index == -1:
            fo_shape = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-shape-anno.txt", "w")
            fo_exp = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-exp-anno.txt", "w")
            fo_pose = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-pose-anno.txt", "w")
            fo_light = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-light-anno.txt", "w")
            fo_cam = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-cam-anno.txt", "w")
            fo_detail = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-detail-anno.txt", "w")
            fo_tform = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-tform-anno.txt", "w")
            fo_albedo = open(f"{args.save_params_folder}/ffhq-{args.params_prefix}-albedo-anno.txt", "w")
            
            fo_dict = {'shape':fo_shape, 'exp':fo_exp, 'pose':fo_pose, 
                    'light':fo_light, 'cam':fo_cam, 'detail':fo_detail,
                    'tform':fo_tform, 'albedo':fo_albedo}
        else:
            fo_shape = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-shape-anno.txt", "w")
            fo_exp = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-exp-anno.txt", "w")
            fo_pose = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-pose-anno.txt", "w")
            fo_light = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-light-anno.txt", "w")
            fo_cam = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-cam-anno.txt", "w")
            fo_detail = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-detail-anno.txt", "w")
            fo_tform = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-tform-anno.txt", "w")
            fo_albedo = open(f"{args.save_params_folder}/ffhq-{start}-{end}-{args.params_prefix}-albedo-anno.txt", "w")
            
            fo_dict = {'shape':fo_shape, 'exp':fo_exp, 'pose':fo_pose, 
                    'light':fo_light, 'cam':fo_cam, 'detail':fo_detail,
                    'tform':fo_tform, 'albedo':fo_albedo}
    #NOTE: SAVEING IMAGES
    clip_path = f"{args.save_images_folder}_wclip/{args.set}"
    woclip_path = f"{args.save_images_folder}_woclip/{args.set}"
    os.makedirs(clip_path, exist_ok=True)
    os.makedirs(woclip_path, exist_ok=True)
    
    for i in tqdm(data_iters):
        # if i == 300:
        #     break
        name = testdata[i]['imagename']
        ext = testdata[i]['imageext']
        images = testdata[i]['image'].to(device)[None,...]
        
        with th.no_grad():
            codedict = deca.encode(images)
            codedict.update({'name':name})
            
            if not args.fast_save_params:
                opdict, visdict = deca.decode(codedict, use_template=args.useTemplate, mean_cam=mean_cam) #tensor
                if args.render_orig:
                    if args.useAvgTform:
                        tform = mean_tform
                    else:
                        tform = testdata[i]['tform'][None, ...]
                    tform_inv = th.inverse(tform).transpose(1,2).to(device)
                    original_image = testdata[i]['original_image'][None, ...].to(device)
                    _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform_inv, use_template=args.useTemplate, mean_cam=mean_cam)    
                    orig_visdict['inputs'] = original_image
                    visdict.update(orig_visdict)
                    
                from torchvision.utils import save_image
                rendered_image = orig_visdict['shape_images']
                rendered_image = rendered_image.permute((0, 2, 3, 1))   # BxHxWxC
                for j in range(rendered_image.shape[0]):
                    np.save(file=f"{woclip_path}/{name}.npy", arr=rendered_image[j].cpu().numpy())
                    save_image(tensor=rendered_image[j].permute((2, 0, 1)).cpu(), fp=f"{clip_path}/{name}.png")
                # save_image(orig_visdict['shape_images'], fp=f"{args.save_images_folder}/{name}.png")
                
        # Params according to fo_dict
        if args.save_params_folder is not None:
            for k, fo in fo_dict.items():
                if k == 'tform':
                    tform = testdata[i]['tform'][None, ...].cpu().numpy().flatten()
                    fo.write(name + f"{ext} ")
                    fo_dict['tform'].write(" ".join([str(x) for x in tform]) + "\n")
                elif k=='albedo':
                    a = codedict['tex'].cpu().numpy().flatten()
                    fo.write(name + f"{ext} ")
                    fo.write(" ".join([str(x) for x in a]) + "\n")
                else:
                    a = codedict[k].cpu().numpy().flatten()
                    fo.write(name + f"{ext} ")
                    fo.write(" ".join([str(x) for x in a]) + "\n")

    print(f'-- please check the results in {args.save_images_folder}')
    
    if args.save_params_folder is not None:
        for k, fo in fo_dict.items():
            fo.close()

# if __name__ == "__main__":
if args.useTemplate and args.useAvgTform and args.useAvgCam:
    mean_face_dict = mean_face()
main()
