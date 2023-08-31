import numpy as np
import matplotlib.pyplot as plt
import torch as th
import warnings
warnings.filterwarnings("ignore") 
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import blobfile as bf
import tqdm
import cv2 
from PIL import Image


class ArcFaceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def get_arcface_emb(img_path, device, arcface_ckpt_path='../../cond_utils/arcface/pretrained/BEST_checkpoint_r18.tar'):

    # Model parameters
    image_w = 112
    image_h = 112
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'normalize': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'none':None
    }

    # loading model
    checkpoint = th.load(arcface_ckpt_path, map_location=device)
    model = checkpoint['model'].module.to(device)

    ## load data for my testing only
    paths = _list_image_files_recursively(img_path)

    images = []
    images_resized = []
    emb_dict = {}
    for i, fn in tqdm.tqdm(enumerate(paths), desc='Open&Resize images'):
        images.append(np.array(Image.open(fn)))
        images_resized.append(cv2.resize(images[i], (image_h, image_w)))
        emb_dict[fn.split('/')[-1]] = None

    # reshape to prefered width height(112, 112)
    images_resized = np.stack(images_resized)
    # create dataset
    dataset = ArcFaceDataset(images_resized, data_transforms['normalize'])
    loader = DataLoader(
        dataset,
        num_workers=24,
        batch_size=1000,
        shuffle=False,
        drop_last=False
    )

    # infer
    model.eval()
    emb = []
    with th.no_grad():
        for i, input_image in tqdm.tqdm(enumerate(loader), desc='Generate Face Embedding'):
            input_image = input_image.to(device)
            features = model(input_image)
            emb.append(features.detach().cpu().numpy())

    emb = np.concatenate(emb, axis=0)
    for i, k in enumerate(emb_dict.keys()):
        emb_dict[k] = {'faceemb':emb[i]}

    return emb_dict, emb
