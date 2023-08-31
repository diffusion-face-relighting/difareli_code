import numpy as np
import cv2

class recolor:
    def rgb_to_sepia(img):
        r = img[..., [0]].copy()
        g = img[..., [1]].copy()
        b = img[..., [2]].copy()
        r_s = (r * 0.393) + (g * 0.769) + (b * 0.189)
        g_s = (r * 0.349) + (g * 0.686) + (b * 0.168)
        b_s = (r * 0.272) + (g * 0.534) + (b * 0.131)

        img_sepia = np.concatenate((r_s, g_s, b_s), axis=2)
        img_sepia = np.clip(img_sepia, 0, 255).astype(np.int)

        return img_sepia

    def rgb_sw_chn(img, ch='rgb'):
        assert len(ch) == 3
        r, g, b = 0, 1 ,2
        ch_n = []
        for c in ch:
            if c == 'r':
                ch_n.append(r)
            elif c == 'g':
                ch_n.append(g)
            elif c == 'b':
                ch_n.append(b)
            else:
                raise NotImplementedError

        img_ = img[..., ch_n].copy()
        return img_

    def rgb_to_hsv(img):
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img_

    def rgb_to_hls(img):
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        return img_

    def rgb_to_ycrcb(img):
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return img_

    def rgb_to_luv(img):
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        return img_

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