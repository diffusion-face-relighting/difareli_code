import get_arcface_emb
import numpy as np
import os
import argparse
import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='Input images', required=True)
parser.add_argument('--set', help='Set of data (e.g. train/valid/test)', required=True)
parser.add_argument('--out_dir', help='Output dir', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    print(f'{os.path.abspath(os.path.dirname(__file__))}/pretrained/BEST_checkpoint_r18.tar')
    arcface_ckpt = f'{os.path.abspath(os.path.dirname(__file__))}/pretrained/BEST_checkpoint_r18.tar'
    out_dict, out = get_arcface_emb.get_arcface_emb(img_path=args.image_dir, device='cuda', arcface_ckpt_path=arcface_ckpt)
    
    os.makedirs(args.out_dir, exist_ok=True)
    fo_emb = open(f"{args.out_dir}/ffhq-{args.set}-faceemb-anno.txt", "w")
    for i, k in enumerate(tqdm.tqdm(out_dict.keys())):
        a = out_dict[k]['faceemb'].flatten()
        assert np.all(np.abs(a-out[i]) < 1e-8)
        fo_emb.write(k + " ")
        fo_emb.write(" ".join([str(x) for x in a]) + "\n")
    fo_emb.close
    