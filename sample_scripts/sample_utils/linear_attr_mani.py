import torch as th
import numpy as np
from . import mani_utils, file_utils

class LinearClassifier(th.nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        self.cls = th.nn.Linear(cfg.param_model.light, 1)
        self.opt = th.optim.Adam(self.parameters())

    def forward(self, x):
        output = self.cls(x)
        return output
    
    def cal_loss(self, gt, pred, weighted_loss):
        if weighted_loss is not None:
            loss_fn = th.nn.BCEWithLogitsLoss(weight=weighted_loss)
        else:
            loss_fn = th.nn.BCEWithLogitsLoss()

        loss = loss_fn(pred, gt)
        return loss

    def train(self, gt, input, n_iters, weighted_loss, progress=True):
        print(f"[#] Training Linear Classifier with iterations={n_iters}, samples_size={gt.shape[0]}")
        if progress:
            import tqdm
            t = tqdm.trange(n_iters, desc="")
        else:
            t = range(n_iters)
        for i in t:
            self.opt.zero_grad()
            pred = self.forward(input)
            loss = self.cal_loss(gt=gt, pred=pred, weighted_loss=weighted_loss)
            loss.backward()
            self.opt.step()
            if i % 500 == 0 and progress:
                t.set_description(f"[#] Loss = {loss}")
                t.refresh() # to show immediately the update

    def evaluate(self, gt, input):
        sigmoid = th.nn.Sigmoid()
        pred = self.forward(input.cuda().float())
        pred = sigmoid(pred) > 0.5
        accuracy = (th.sum(pred == gt.cuda()) / pred.shape[0]) * 100
        print(f"[#] Accuracy = {accuracy}")

def distance(a, b, dist_type='l2'):
    if dist_type == 'l1':
        return np.sum(np.abs(a-b))
    elif dist_type == 'l2':
        return np.sum((a-b)**2)

def get_weighted_average(dist_arr, sigma):
    weighted_term = 1/np.sqrt(2 * np.pi * (sigma**2))* (np.exp(-np.power(dist_arr, 2.) / (2 * np.power(sigma, 2.))))
    # import matplotlib.pyplot as plt
    # plt.plot(weighted_term)
    # plt.show()
    # plt.plot(dist_arr)
    # plt.show()
    return weighted_term

def retrieve_topk_params(params_set, ref_params, cfg, img_dataset_path, sigma, k, dist_type='l2'):
    '''
    Return images and params of top-k nearest parameters
    :param params_set: parameters in dict-like e.g. {'0.jpg':{'light' : ..., 'pose' : ..., ...}}
    :param ref_params: reference parameters in numpy array
    :param cfg: config file (only use in loading image function)
    :param img_dataset_path: datapath to image
    :param dist_type: distance type used in calculate the top-k nearest
    :param k: Number of nearest-neighbour
    '''

    light_dist = []
    img_name_list = []
    for img_name in params_set.keys():
        light_dist.append(distance(params_set[img_name]['light'], ref_params, dist_type=dist_type))
        img_name_list.append(img_name)


    min_idx = np.argsort(light_dist)[:k]
    weighted_term = get_weighted_average(np.array(light_dist)[min_idx], sigma=sigma)

    img_path = file_utils._list_image_files_recursively(img_dataset_path)
    img_path = [img_path[i] for i in min_idx]
    img_name = [path.split('/')[-1] for path in img_path]

    images = mani_utils.load_image(all_path=img_path, cfg=cfg, vis=True)['image']
    params_filtered = {i:params_set[i] for i in img_name}
    return images, params_filtered, th.tensor(weighted_term)