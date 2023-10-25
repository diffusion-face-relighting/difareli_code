from templates import *
from templates_latent import *
import argparse, pickle, os, glob
import blobfile as bf

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='./output_shadow_score')
parser.add_argument('--shadow_weight', type=str, default='./shadow_cls_weight.pkl')
parser.add_argument('--set_', type=str, default='valid')
args = parser.parse_args()

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

if __name__ == '__main__':
    
    gpus = [0]
    conf = ffhq256_autoenc()
    conf.data_name = 'custom'
    conf.shadow_input_path = args.image_dir
    conf.shadow_output_path = args.out_dir
    # im_name = sorted(glob.glob(os.path.join(args.image_dir, '*jpg')))
    im_name = _list_image_files_recursively(args.image_dir)
    print(len(im_name))
    
    conf.eval_programs = ['infer', 'shadow']
    train(conf, gpus=gpus, mode='eval')

    lat = torch.load(os.path.join(args.out_dir, 'latent.pkl'), map_location='cpu')

    pkl_file = open(args.shadow_weight, 'rb')
    cls_weight = pickle.load(pkl_file)
    ndata = lat['conds'].numpy()
    proj_data = ndata@(cls_weight['weight'].T)

    # if not os.path.exists(os.path.dirname(args.outpath)):
    #     os.makedirs(os.path.dirname(args.outpath))
    # with open(os.path.join(args.outpath, 'shadow.txt'), 'w') as f:
    os.makedirs(f'{args.out_dir}', exist_ok=True)
    with open(f'{args.out_dir}/ffhq-{args.set_}-shadow-anno.txt', 'w') as f:
        for i in range(ndata.shape[0]):
            m = im_name[i].split('/')[-1]
            f.write(f'{m} {proj_data[i].item()}')
            f.write('\n')