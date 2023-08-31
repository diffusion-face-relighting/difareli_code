import numpy as np
import torch as th

class Renderer():
    def __init__(self, num_SH, reduce_shading):
        ## SH factors for lighting
        pi = np.pi
        self.reduce_shading = reduce_shading
        self.num_SH = num_SH
        self.constant_factor = th.tensor([1/np.sqrt(4*pi), 
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))), 
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))),
                                        ((2*pi)/3)*(np.sqrt(3/(4*pi))), 
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))),
                                        (pi/4)*(3)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), 
                                        (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.constant_factor = self.constant_factor[:self.num_SH]
        if self.reduce_shading:
            self.shading_channels = 3
        else:
            self.shading_channels = int(self.num_SH * 3)

    def add_SHlight(self, normal_images, sh_coeff):
        """
        Apply the SH to normals(features map)
        :param h: normals(features map) in [B x #C_Normals x H xW]
        :param sh_coeff: SH lighting condition in [B x 27]
        :param reduce: 
            - True : reducing the added SH in to shaded image [B x 3 x H x W]
            - False : Non-reducing into shaded image [B x 27 x H x W]
        """
        N = normal_images
        assert sh_coeff.shape == (N.shape[0], self.num_SH * 3)
        sh_coeff = sh_coeff.reshape(N.shape[0], self.num_SH, 3)
        sh = th.stack([
                N[:,0]*0.+1.,   # 1
                N[:,0],         # X
                N[:,1],         # Y
                N[:,2],         # Z
                N[:,0]*N[:,1],  # X*Y
                N[:,0]*N[:,2],  # X*Z
                N[:,1]*N[:,2],  # Y*Z
                N[:,0]**2 - N[:,1]**2,  # X**2 - Y**2
                3*(N[:,2]**2) - 1,      # 3(Z**2) - 1
                ], 
                1) # [bz, 9, h, w]
        
        sh = sh[:, :self.num_SH, :, :]
        sh = sh.type_as(N) * self.constant_factor[None, :, None, None].type_as(N)
        if self.reduce_shading:
            shading = th.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)
        else:
            shading = sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :]
            shading = th.flatten(shading, start_dim=1, end_dim=2)

        return shading