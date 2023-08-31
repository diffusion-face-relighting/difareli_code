import glob
from config.base_config import parse_args
import torch as th
from guided_diffusion.script_util import (
    create_img_and_diffusion,
)

class CkptLoader():
    def __init__(self, log_dir, cfg_name, device=None) -> None:
        self.sshfs_mount_path = "/data/mint/model_logs_mount/"
        self.sshfs_path = "/data/mint/model_logs/"

        self.log_dir = log_dir
        self.cfg_name = cfg_name
        self.model_path = self.get_model_path()
        self.cfg = self.get_cfg()
        self.name = self.cfg.img_model.name
        if device is not None:
            self.device = device
        else:
            if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0:
               self.device = 'cuda' 
            else : self.device = 'cpu'
            
                

    # Config file
    def get_cfg(self):
        cfg_file_path = glob.glob("/home/mint/guided-diffusion/config/**", recursive=True)
        cfg_file_path = [cfg_path for cfg_path in cfg_file_path if f"/{self.cfg_name}" in cfg_path]    # Add /{}/ to achieve a case-sensitive of folder
        print("[#] Config Path : ", cfg_file_path)
        assert len(cfg_file_path) <= 1
        assert len(cfg_file_path) > 0
        cfg_file = cfg_file_path[0]
        cfg = parse_args(ipynb={'mode':True, 'cfg':cfg_file})
        return cfg

    # Log & Checkpoint file 
    def get_model_path(self,):
        model_logs_path = glob.glob(f"{self.sshfs_mount_path}/*/*/", recursive=True) + glob.glob(f"{self.sshfs_path}/*/", recursive=True)
        model_path = [m_log for m_log in model_logs_path if f"/{self.log_dir}/" in m_log]    # Add /{}/ to achieve a case-sensitive of folder
        print("[#] Model Path : ")
        for i in range(len(model_path)):
            print(f"#{i} : {model_path[i]}")
        # assert (len(model_path) <= 1 and len(model_path) > 0)
        if len(model_path) > 1:
            sel = int(input("[#] Please pick specific model index : "))
            return model_path[sel]
        else:
            return model_path[0]

    # Load model
    def load_model(self, ckpt_selector, step):
        if ckpt_selector == "ema":
            ckpt = f"ema_0.9999_{step}"
        elif ckpt_selector == "model":
            ckpt = f"model{step}"
        else: raise NotImplementedError

        self.available_model()
        # self.cfg.diffusion.diffusion_steps = 25
        model_dict, diffusion = create_img_and_diffusion(self.cfg)
        model_dict = {k: v for k, v in model_dict.items() if v is not None}
        for m_name in model_dict.keys():
            model_path = f"{self.model_path}/{m_name}_{ckpt}.pt"
            print(f"[#] Loading...{model_path}")

            model_dict[m_name].load_state_dict(
                th.load(model_path, map_location="cpu")
            )
            model_dict[m_name].to(self.device)
            model_dict[m_name].eval()

        return model_dict, diffusion

    def available_model(self):
        import re
        avail_m = glob.glob(f"{self.model_path}/*.pt")
        filtered_m = []
        for m in avail_m:
            r = re.search(r"(_(\d+).pt)", m)
            if r:
                filtered_m.append(list(r.groups())[0])
        print("[#] Available ckpt : ", sorted(filtered_m))
