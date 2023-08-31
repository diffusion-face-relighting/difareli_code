import argparse
import subprocess
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='Input images', required=True)
parser.add_argument('--out_dataset_dir', help='The Dataset name to be output.', required=True)
parser.add_argument('--set', help='set e.g. [train, valid, test]', default='valid')
parser.add_argument('--ffhq_align', action='store_true', default=False)
parser.add_argument('--faceseg', action='store_true', default=False)
parser.add_argument('--deca', action='store_true', default=False)
parser.add_argument('--arcface', action='store_true', default=False)
parser.add_argument('--shadow', action='store_true', default=False)
args = parser.parse_args()

if __name__ == '__main__':
    print(f"[#] Creating the dataset : {args.out_dataset_dir}")
    print(f"[#] Image dir : {args.image_dir}")
    sep_str = "#"*150
    
    if args.ffhq_align:
        #NOTE: Running the ffhq align
        '''
        python align.py -i ../../ITW/itw_images -o ../../ITW/itw_images_aligned
        '''
        print(sep_str)
        print("\tCurrent directory : ", os.getcwd())
        os.chdir('./FFHQ_align/')
        print("\tStepping into FFHQ_align : ", os.getcwd())
        print("[#] Creating the FFHQ alignment images...")
        env =  "/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/FFHQ_align/ffhq_env/bin/python"
        save_aligned = os.path.basename(os.path.normpath(args.image_dir)) + "_aligned"
        print(' '.join([env, './align.py', 
                        '-i', args.image_dir, 
                        '-o', f'{args.out_dataset_dir}/{save_aligned}/{args.set}',
                      ]))
        subprocess.call([env, './align.py', 
                        '-i', args.image_dir, 
                        '-o', f'{args.out_dataset_dir}/{save_aligned}/{args.set}',
                        ])
        
        os.chdir('../')
        print("\tReturn to parent : ", os.getcwd())
        print(sep_str)
        
        print(sep_str)
        print("[#] Switching from ITW images to FFHQ's aligned images...")
        args.image_dir = f'{args.out_dataset_dir}/{save_aligned}/{args.set}/'
        print(sep_str)
        
    if args.faceseg:
        #NOTE: Running Face segment on 
        '''
        python inference.py --data ../../ITW/itw_images --output_path ../../face_segment/
        '''
        print(sep_str)
        print("[#] Creating the Face segment (face-parsing.PyTorch)...")
        print(f"[#] From directory : {args.image_dir}")
        env =  "/home/mint/miniconda3/envs/dpm_create_dataset/bin/python"
        subprocess.call([env, './face-parsing.PyTorch/inference.py', 
                        '--data', args.image_dir, 
                        '--output_path', f'{args.out_dataset_dir}/face_segment/{args.set}',
                        ])
        print(sep_str)
    
    if args.deca:
        #NOTE: Running DEC
        '''
        python estimate_deca_for_dpm.py --useTex True --render_orig True 
        --inputpath ../../../ITW/itw_images/  --useTemplate False --useAvgCam False 
        --useAvgTform False --save_images_folder ../../../ITW/rendered_images/decaface_images 
        --set valid --fast_save_params False --save_params_folder ../../../ITW/params/valid
        '''
        print(sep_str)
        print("[#] Creating the Face rendering (DECA)...")
        print(f"[#] From directory : {args.image_dir}")
        print("\tCurrent directory : ", os.getcwd())
        os.chdir('./DECA/script')
        print("\tStepping into DECA-dir : ", os.getcwd())
        env =  "/home/mint/miniconda3/envs/dpm_create_dataset/bin/python"
        subprocess.call([env, './estimate_deca_for_dpm.py',
                        '--useTex', 'True',
                        '--inputpath', args.image_dir,
                        '--useTemplate', 'False', 
                        '--useAvgCam', 'False',
                        '--useAvgTform', 'False',
                        '--save_images_folder', f'{args.out_dataset_dir}/rendered_images/deca_masked_face_images',
                        '--set', args.set,
                        '--params_prefix', args.set,
                        '--save_params_folder', f'{args.out_dataset_dir}/params/{args.set}',
                        '--masking_flame'
                        ])
        
        os.chdir('../../')
        print("\tReturn to parent : ", os.getcwd())
        print(sep_str)
        
    if args.arcface:
        #NOTE: Running Arcface
        '''
        python inference.py --image_dir ../../ITW/itw_images --set valid --out_dir ../../ITW/params/valid/
        '''
        print(sep_str)
        print("[#] Creating the Face embedding (Arcface)...")
        print(f"[#] From directory : {args.image_dir}")
        env =  "/home/mint/miniconda3/envs/dpm_create_dataset/bin/python"
        subprocess.call([env, './Arcface/inference.py',
                        '--image_dir', args.image_dir, 
                        '--set', args.set,
                        '--out_dir', f'{args.out_dataset_dir}/params/{args.set}',
                        ])
        print(sep_str)
        
    if args.shadow:
        '''
        python gen_shadow.py -i test_shadow -o ./shadow_out
        '''
        print(sep_str)
        print("[#] Creating the Shadow embedding (DiffAE-Shadow)...")
        print(f"[#] From directory : {args.image_dir}")
        print("\tCurrent directory : ", os.getcwd())
        os.chdir('./diffae/')
        print("\tStepping into DiffAE-dir : ", os.getcwd())
        env =  "/home/mint/miniconda3/envs/dpm_create_dataset/bin/python"
        subprocess.call([env, './gen_shadow.py',
                        '--image_dir', args.image_dir, 
                        '--set_', args.set,
                        '--out_dir', f'{args.out_dataset_dir}/params/{args.set}/',
                        ])
        
        os.chdir('../')
        print("\tReturn to parent : ", os.getcwd())
        print(sep_str)
        
    

# conda activate dpm_create_dataset
# python inference.py --data ../examples/ --output_path ../examples_align/
