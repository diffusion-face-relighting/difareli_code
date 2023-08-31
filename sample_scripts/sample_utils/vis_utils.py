from guided_diffusion.dataloader.img_util import decolor
import matplotlib.pyplot as plt
import torch as th
import torchvision
import numpy as np
import sys, os
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import params_utils as params_utils
import cv2

def plot_sample(img, highlight=None, **kwargs):
    columns = 6
    rows = 17
    fig = plt.figure(figsize=(20, 20), dpi=100)
    img = img.permute(0, 2, 3, 1) # BxHxWxC
    pt = 0
    for i in range(0, img.shape[0]):
        s_ = decolor(s=img[i], out_c='rgb')
        s_ = s_.detach().cpu().numpy()
        
        if highlight is not None:
            top, bottom, left, right = [10]*4

            if i == highlight['base_idx']:
                s_ = cv2.copyMakeBorder(s_, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            elif i == highlight['src_idx']:
                s_ = cv2.copyMakeBorder(s_, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            elif i == highlight['dst_idx']:
                s_ = cv2.copyMakeBorder(s_, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 0, 0))
            
        fig.add_subplot(rows, columns, pt+1)
        plt.imshow(s_)
        pt += 1

        if kwargs is not None:
            # Plot other images
            for k in kwargs:
                fig.add_subplot(rows, columns, pt+1)
                s_ = decolor(s=kwargs[k][i].permute(1, 2, 0), out_c='rgb')
                s_ = s_.detach().cpu().numpy().astype(np.uint8)
                plt.imshow(s_)
                pt += 1
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.65, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.2)
    plt.show()
    return fig

def plot_deca(sample, min_value, max_value, cfg):
    img_ = []
    from tqdm.auto import tqdm
    for i in tqdm(range(sample['deca_output'].shape[0])):
        deca_params = sample['deca_output'][i].clone()
        deca_params = params_utils.denormalize(deca_params, min_val=th.tensor(min_value).cuda(), max_val=th.tensor(max_value).cuda(), a=-cfg.param_model.bound, b=cfg.param_model.bound).float() 
        shape = deca_params[None, :100]
        pose = deca_params[None, 100:106]
        exp = deca_params[None, 106:156]
        cam = deca_params[None, 156:]
        img = params_utils.params_to_model(shape=shape, exp=exp, pose=pose, cam=cam, i=i)
        img_.append(img["shape_images"])

    plot_sample(th.cat(img_, dim=0))
    return th.cat(img_, dim=0)

def plot_image(img, c_len=[], fn='./temp', range="-1to1"):
    """
    :param img: image tensor in B x C x H x W
    """
    import torchvision
    print("Image shape : ", img.shape)
    if c_len == []:
        c_len = list(range(0, img.shape[1], 3)) + [img.shape[1]]
    print("Channel length : ", c_len)
    if img.shape[0] == 1:
        img_plot = []
        for i, c in enumerate(c_len):
            img_plot.append(img[0, 0:c, ...])
            img = img[:, c:, ...]   # Slice out the plotted one
            if range == "-1to1":
                img_plot[i] = ((img_plot[i] + 1) * 127.5) / 255.0
        img_plot = th.stack(img_plot, dim=0)
        torchvision.utils.save_image(tensor=img_plot, fp=f"./{fn}.png")
    else:
        for i, c in enumerate(c_len):
            img_plot = img[:, 0:c, ...]
            img = img[:, c:, ...]   # Slice out the plotted one
            if range == "-1to1":
                img_plot = ((img_plot + 1) * 127.5) / 255.0
            torchvision.utils.save_image(tensor=img_plot, fp=f"./{fn}_{i}.png")

def save_video(fn, frames, fps=30):
    """
    save the video
    Args:
        frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
        fn : path + filename to save
        fps : video fps
    """
    if frames.is_cuda:
        frames = frames.cpu()
    torchvision.io.write_video(video_array=frames, filename=fn, fps=fps)

def save_images(path, fn, frames):
    """
    save the images
    Args:
        frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
        path : save path
        fn : filename to save
    """
    
    for i in range(frames.shape[0]):
        frame = frames[i].cpu().detach()
        torchvision.utils.save_image(tensor=(frame), fp=f"{path}/{fn}_frame{i}.png")
        
def save_images_with_fn(path, fn, frames, fn_list):
    """
    save the images
    Args:
        frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
        path : save path
        fn : filename to save
    """
    
    for i in range(frames.shape[0]):
        frame = frames[i].cpu().detach()
        frame_idx = fn_list[i].split('/')[-1].split('.')[0]
        torchvision.utils.save_image(tensor=(frame), fp=f"{path}/{fn}_{frame_idx}.png")

def save_intermediate(path, out, proc, image_name, bound):
    
    for itmd in out['intermediate']:
        t = itmd['t']
        sample = itmd['sample'].cpu().detach()
        sample = convert2rgb(sample, bound) / 255.0
        pred_xstart = itmd['pred_xstart'].cpu().detach()
        pred_xstart = convert2rgb(pred_xstart, bound) / 255.0
        assert sample.shape[0] == pred_xstart.shape[0]

        batch_size = sample.shape[0]
        for b in range(batch_size):
            pred_xstart_path = f"{path}/{proc}/{image_name[b]}/pred_xstart/"
            sample_path = f"{path}/{proc}/{image_name[b]}/sample/"
            os.makedirs(sample_path, exist_ok=True)
            os.makedirs(pred_xstart_path, exist_ok=True)
            torchvision.utils.save_image(tensor=(sample[[b]]), fp=f"{sample_path}/sample_frame{t[b]}.png")
            torchvision.utils.save_image(tensor=(pred_xstart[[b]]), fp=f"{pred_xstart_path}/pred_xstart_frame{t[b]}.png")

    final_output = convert2rgb(out['final_output']['sample'].cpu().detach(), bound) / 255.0
    for b in range(batch_size):
        final_output_path = f"{path}/{proc}/{image_name[b]}/final_output/"
        os.makedirs(final_output_path, exist_ok=True)
        torchvision.utils.save_image(tensor=(final_output[[b]]), fp=f"{final_output_path}/final_output_frame0.png")
    
def convert2rgb(img, bound):
    """Convert the image from +-bound into 0-255 rgb

    Args:
        img (tensor): input image
        bound (float): bounding value e.g. 1, 0.5, ...

    Returns:
        convert image (tensor) : 
    """
    if bound == 1.0:
        convert_img = (img + 1) * 127.5
    elif bound == 0.5:
        convert_img = (img + 0.5) * 255.0
    return convert_img

def spiralOrder(m, n):
    
    rowStart, rowEnd = 0, m - 1
    colStart, colEnd = 0, n - 1
    idx = []
    while rowStart < rowEnd and colStart < colEnd:
        for i in range(colStart, colEnd + 1):
            idx.append([rowStart, i])
        rowStart += 1

        for i in range(rowStart, rowEnd + 1):
            idx.append([i, colEnd])
        colEnd -= 1

        if rowStart < rowEnd:
            for i in range(colEnd, colStart - 1, -1):
                idx.append([rowEnd, i])
            rowEnd -= 1

        if colStart < colEnd:
            for i in range(rowEnd, rowStart - 1, -1):
                idx.append([i, colStart])
            colStart += 1
    return idx