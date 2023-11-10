# [DiFaReli: Diffusion Face Relighting](https://diffusion-face-relighting.github.io/) (ICCV 2023)
Puntawat Ponglertnapakorn, Nontawat Tritrong, [Supasorn Suwajanakorn](https://www.supasorn.com/)



![Alt text](./misc_md/teaser.png)

## Overview

This repository contains:

1. Preprocessing script for any images
2. Script for training and inference

## Dependencies
1. [DECA](https://github.com/yfeng95/DECA)
2. [Arcface](https://github.com/foamliu/InsightFace-v2/tree/e07b738adecb69b81ac9b8750db964cee673e175)
3. [Face parsing](https://github.com/zllrunning/face-parsing.PyTorch)
4. [Diff-AE](https://github.com/phizaz/diffae)

We also provided the requirements.txt for the dependencies. You can install all dependencies by running `pip install -r <req_file>.txt`

## Quick start
1. Clone & Install all dependencies
2. Prepare your images
3. Run the preprocessing script
4. Run the training or inference script

## Dependencies Installation
1. For basic installation, you can follow the instruction in each repository given follow.
2. For [DECA](https://github.com/yfeng95/DECA)

    2.1 Masking out the render face: Since we masking out some part of the face mesh (e.g. ear, neck, etc.). We use additional [FLAME_masks_face-id.pkl](./sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl) file. Make sure you change the path at [this line](./sample_scripts/sample_utils/params_utils.py#L123).

    2.2 Make sure you change all of the path in sample_utils directory to match your path (e.g [params_utils.py](./sample_scripts/sample_utils/params_utils.py#L136), etc.).

## Preprocessing

1. Create and put your images in the folder following this structure:
```
./test_images
└── images
    └── valid
        ├── 60922.jpg
        └── 63252.jpg
```

2. Align the images:\
Command: `python align.py -i <path_to_image> -o <output_path>`\
Example: `python align.py -i ../test_images/images/valid -o ../test_images/aligned_images/valid`

The output will be in the following structure:
```
./test_images
├── aligned_images <--- Aligned Output 
│   └── valid
│       ├── 60922.png
│       └── 63252.png
└── valid
    ├── 60922.jpg
    └── 63252.jpg
```

3. Preprocessing the images:

Compute face parsing, Deca face estimation, Arcface feature extraction and Diff-AE's degree of shadow. We separate into 2 main steps as follows:

3.1. Clone & Install [DECA](https://github.com/yfeng95/DECA), [Diff-AE](https://github.com/phizaz/diffae), [Face parsing](https://github.com/zllrunning/face-parsing.PyTorch) and [Arcface](https://github.com/foamliu/InsightFace-v2/tree/e07b738adecb69b81ac9b8750db964cee673e175).Then put everything into the Relighting_preprocessing_tools folder. The folder structure should look like this:
```
.
└── Relighting_preprocessing_tools
    ├── Arcface                   <--- Arcface
    ├── DECA                      <--- DECA
    ├── diffae                    <--- DiffAE
    ├── face-parsing.PyTorch      <--- Face parsing
    ├── FFHQ_align                <--- Image alignment
    ├── create_dataset.py
    └── test_images
```

3.2 Running a preprocessing script:\
Command: `python create_dataset.py --image_dir <path_to_images> --out_dataset_dir <output_path> --faceseg --deca --arcface --shadow`\
Example: `python create_dataset.py --image_dir /home/user/difareli_code/preprocess_scripts/Relighting_preprocessing_tools/test_images/aligned_images/valid/ --out_dataset_dir ./test_images/ --faceseg --deca --arcface --shadow`

Your final output folder should look like [this](./misc_md/preprocess_out.md).\
[#Note] The path here needs to be aboslute path to the input folder.

## Training model
We provide the script and the config file for training our difareli.

Command:`python train_scripts/image_train.py --train.log_dir <ckpt_savepath> --train.batch_size <batch_size> --train.n_gpus <n_gpus> --cfg <path_to_cfg>`\
Example:`python train_scripts/image_train.py --train.log_dir /data/mint/model_logs/test --train.batch_size 32 --train.n_gpus 4 --cfg ./config/difareli/difareli_128.yaml`

[#] There's some other arguments that you can use to train the model. The command line arguments will override the config file and applied in training.\
For example, if you want to change the number of sampling interval, you can use `--train.sampling_interval <numbers_of_sampling_interval>`. and this will regards whatever in the config file as default and override it with the new value.

## Checkpoints
We provide checkpoints for the following models:
1. [difareli_128](https://vistec-my.sharepoint.com/:f:/g/personal/puntawat_p_s19_vistec_ac_th/EsrzPdKduKhFvnNXiT3XMmUBcRBvnfNlyhCS6JJM1r0qrw?e=BJO18a) (128x128 resolution)
2. [difareli_256](https://vistec-my.sharepoint.com/:f:/g/personal/puntawat_p_s19_vistec_ac_th/EhO-rlrfEYpPm8dk6AtH-2cB8N7G_O8wm1Q_vLBVhr43Dw?e=3rlCvG) (256x256 resolution)

You can download and put it in any folder. <strong>Don't forget</strong> to update the path to match the location where you placed the checkpoint at this [line](https://github.com/diffusion-face-relighting/difareli_code/blob/main/sample_scripts/sample_utils/ckpt_utils.py#L10). For example, if you put the checkpoint in `./model_logs/difareli_256/`, you need to change the path to `./model_logs/`.

## Inference
1. To inference our difareli, you need to prepare the sample pair in json format (e.g. [shadow.json](./sample_scripts/inference/reshadow.json) or [relight.json](./sample_scripts/inference/relight.json))
2. Run the inference script

### Relighting 
To relight the image, you can to run the following command:\
Command:`python relight.py --dataset <ffhq/mp/etc.> --set <train/valid> --step <ckpt_step> --out_dir <sampling_output> --cfg_name <cfg_name>.yaml --log_dir <ckpt_savename> --diffusion_steps <1000> --timestep_respacing <250/500/1000,etc> --sample_pair_json <path_to_sample_file> --sample_pair_mode pair --itp render_face --itp_step <number_of_frames> --batch_size <batch_size> --gpu_id <gpu_id> --lerp --idx <start_idx> <end_idx>`

Example:`python relight.py --dataset ffhq --set valid --step 085000 --out_dir ./output_relight/ --cfg_name difareli_256.yaml --log_dir difareli_256 --diffusion_steps 1000 --timestep_respacing 1000 --sample_pair_json ./relight.json --sample_pair_mode pair --itp render_face --itp_step 5 --batch_size 1 --gpu_id 0 --lerp --idx 0 10`

This will generate the relighting frame and video in the output folder.

[#] There's some other arguments that you can use to inference the model. You can refers to the argument parser in the [script](./sample_scripts/inference/relight.py) for more details.

### Reshadow
1. Reshadow using arbitrary 'c' value:

Command:`python relight.py --dataset <ffhq/mp/etc.> --set <train/valid> --step <ckpt_step> --out_dir <sampling_output> --cfg_name <cfg_name>.yaml --log_dir <ckpt_savename> --diffusion_steps <1000> --timestep_respacing <250/500/1000,etc> --sample_pair_json <path_to_sample_file> --sample_pair_mode pair --itp shadow --itp_step 10 --batch_size 10 --gpu_id 2 --lerp --save_vid --idx 0 24 --vary_shadow_range <min_c> <max_c>`

Example:`python reshadow.py --dataset ffhq --set valid --step 085000 --out_dir ./out_reshadow/ --cfg_name relight_256.yaml --log_dir relight_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./reshadow.json --sample_pair_mode pair --itp shadow --itp_step 10 --batch_size 10 --gpu_id 2 --lerp --save_vid --idx 0 24 --vary_shadow_range -5 10`

2. Reshadow using the shadow value specified in the sample json file:\
You can change from `--vary_shadow_range <min_c> <max_c>` into `--vary_shadow` and the script will use the shadow value specified in the sample json file.

[#] There's some other arguments that you can use to inference the model. You can refers to the argument parser in the [script](./sample_scripts/inference/reshadow.py) for more details.

## Citation
If you find this code useful for your research, please cite our paper:
```
@InProceedings{ponglertnapakorn2023difareli,
  title={DiFaReli: Diffusion Face Relighting},
  author={Ponglertnapakorn, Puntawat and Tritrong, Nontawat and Suwajanakorn, Supasorn},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```