import numpy as np
import math
import PIL, cv2
import random
import torch as th
import blobfile as bf
import os
import torchvision

from sample_scripts.cond_utils.arcface.config_arcface import IMG_DIR

def resize_arr(pil_image, image_size):
    img = pil_image.resize((image_size, image_size), PIL.Image.ANTIALIAS)
    return np.array(img)

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def decolor(s, out_c='rgb'):
    if out_c in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c == 'luv':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c == 'ycrcb':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c in ['hsv', 'hls']:
        h = (s[..., [0]] + 1) * 90.0 
        l_s = (s[..., [1]] + 1) * 127.5
        v = (s[..., [2]] + 1) * 127.5
        s_ = th.cat((h, l_s, v), axis=2).clamp(0, 255).to(th.uint8)
    elif out_c == 'sepia':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)

    else: raise NotImplementedError

    return s_

def augmentation(pil_image, cfg):
    # Resize image by resizing/cropping to match the resolution
    if cfg.img_model.resize_mode == 'random_crop':
        arr = random_crop_arr(pil_image, cfg.img_model.image_size)
    elif cfg.img_model.resize_mode == 'center_crop':
        arr = center_crop_arr(pil_image, cfg.img_model.image_size)
    elif cfg.img_model.resize_mode == 'resize':
        arr = resize_arr(pil_image, cfg.img_model.image_size)
    else: raise NotImplemented

    return arr

def prep_images(path, image_size):
    '''
    Preprocess the image
    '''
    with bf.BlobFile(path, "rb") as f:
        pil_image = PIL.Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")

    raw_img = resize_arr(pil_image=pil_image, image_size=image_size)
    raw_img = (raw_img / 127.5) - 1

    return np.transpose(raw_img, [2, 0, 1])

def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

def sequence2gif(imgs, img_size, save_path='./animated_results', save_fn=''):
    os.makedirs(name=save_path, exist_ok=True)
    out_fn = f'./{save_path}/{save_fn}.gif'
    gif = []
    for img in imgs:
        img = np.array(img)
        img = ((img + 1) * 127.5).astype(np.uint8)
        if img.shape == (3, img_size, img_size):
            img = np.transpose(img, (1, 2, 0))
        gif.append(PIL.Image.fromarray(img))
    gif[0].save(out_fn, save_all=True, append_images=gif[1:])

def sequence2video(img_path, save_path, vid_name, img_size):
    os.makedirs(save_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    video = cv2.VideoWriter(f"{save_path}/{vid_name}.avi", fourcc, 1, (img_size, img_size)) 
  
    import tqdm
    # Appending the images to the video one by one
    for image in tqdm.tqdm(img_path): 
        frame = cv2.imread(image)
        frame = cv2.resize(frame, (img_size, img_size))
        video.write(frame)
      
    video.release()  # releasing the video generated

def sequence2video(imgs, img_size, save_path='./animated_results/', save_fn=''):
    os.makedirs(name=save_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{save_path}/{save_fn}.mp4", fourcc, 5, (img_size, img_size)) 
  
    # Appending the images to the video one by one
    for img in imgs:
        img = np.array(img)
        img = ((img + 1) * 127.5).astype(np.uint8)
        if img.shape == (3, img_size, img_size):
            img = np.transpose(img, (1, 2, 0))
        video.write(img[..., ::-1])
      
    video.release()  # releasing the video generated

def blur(raw_img, sigma):
    ksize = int(raw_img.shape[0] * 0.1)
    ksize = ksize if ksize % 2 != 0 else ksize+1
    blur_kernel = torchvision.transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)
    raw_img = raw_img.permute(dims=(2, 0, 1))
    blur_img = blur_kernel(raw_img)
    return blur_img