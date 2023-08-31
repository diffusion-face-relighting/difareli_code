#!/usr/bin/python
# -*- encoding: utf-8 -*-

from pkg_resources import require
from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    # atts = [0 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
    # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_im = cv2.resize(vis_im, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255



    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        im_name = save_path[:-4].split('/')[-1]
        anno_path = f'{args.output_path}/anno/'
        vis_path = f'{args.output_path}/vis/'
        os.makedirs(anno_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        cv2.imwrite(anno_path + f'anno_{im_name}.png', vis_parsing_anno)

        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        bg = (vis_parsing_anno == 0)[..., None]
        skin = (vis_parsing_anno == 1)[..., None]
        l_brow = (vis_parsing_anno == 2)[..., None]
        r_brow = (vis_parsing_anno == 3)[..., None]
        l_eye = (vis_parsing_anno == 4)[..., None]
        r_eye = (vis_parsing_anno == 5)[..., None]
        eye_g = (vis_parsing_anno == 6)[..., None]
        l_ear = (vis_parsing_anno == 7)[..., None]
        r_ear = (vis_parsing_anno == 8)[..., None]
        ear_r = (vis_parsing_anno == 9)[..., None]
        nose = (vis_parsing_anno == 10)[..., None]
        mouth = (vis_parsing_anno == 11)[..., None]
        u_lip = (vis_parsing_anno == 12)[..., None]
        l_lip = (vis_parsing_anno == 13)[..., None]
        neck = (vis_parsing_anno == 14)[..., None]
        neck_l = (vis_parsing_anno == 15)[..., None]

        face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, neck_l))
        vis_parsing_anno = np.stack([vis_parsing_anno]*3, axis=-1)
        outer_face = ~face
        combined_im = np.concatenate((
            vis_parsing_anno, 
            vis_im, 
            im[:, :, ::-1] * face, 
            im[:, :, ::-1] * outer_face, 
            im[:, :, ::-1] * bg, 
            im[:, :, ::-1] * ~bg, 
            im[:, :, ::-1]), axis=1)

        cv2.imwrite(vis_path + f'res_{im_name}.png', combined_im)


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    import tqdm
    with torch.no_grad():
        for image_path in tqdm.tqdm(os.listdir(dspth)):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth=args.data, cp=os.path.join(f'{os.path.abspath(os.path.dirname(__file__))}/res/cp', '79999_iter.pth'))


