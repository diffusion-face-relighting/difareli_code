import torch as th
import copy

def make_deepcopyable(model_kwargs, keys):
    '''
    Make the dict-like to be used with deepcopy function
    :param model_kwargs: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''
    for key in keys:
        if key in ['image_name', 'raw_image_path', 'cfg', 'use_render_itp', 'use_cond_xt_fn']:
            continue
        else:
            if th.is_tensor(model_kwargs[key]):
                model_kwargs[key] = model_kwargs[key].detach()
            elif isinstance(model_kwargs[key], list):
                for i in range(len(model_kwargs[key])):
                    model_kwargs[key][i] = model_kwargs[key][i].detach()

    model_kwargs_copy = copy.deepcopy(model_kwargs)
    return model_kwargs_copy

def dict_type_as(in_d, target_d, keys):
    '''
    Apply type_as() of the dict-like.
    :param in_d: a dict-like with {'key1': tensor, ...}
    :param target_d: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''
    for key in keys:
        if key in ['image_name', 'raw_image_path']:
            continue
        else:
            if th.is_tensor(in_d[key]):
                in_d[key] = in_d[key].type_as(target_d[key])
            elif isinstance(in_d[key], list):
                for i in range(len(in_d[key])):
                    in_d[key][i] = in_d[key][i].type_as(target_d[key][i])
                    
    return in_d

def dict_detach(in_d, keys):
    '''
    Apply type_as() of the dict-like.
    :param in_d: a dict-like with {'key1': tensor, ...}
    :param target_d: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''
    for key in keys:
        if key in ['image_name', 'raw_image_path']:
            continue
        else:
            if th.is_tensor(in_d[key]):
                in_d[key] = in_d[key].detach()
            elif isinstance(in_d[key], list):
                for i in range(len(in_d[key])):
                    in_d[key][i] = in_d[key][i].detach()
                    
    return in_d


def dict_slice(in_d, keys, n):
    '''
    Apply type_as() of the dict-like.
    **** Every tensor must be in batch size (B x ...) ****
    :param in_d: a dict-like with {'key1': tensor, ...}
    :param target_d: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''
    for key in keys:
        if key in ['image_name', 'raw_image_path']:
            continue
        else:
            if th.is_tensor(in_d[key]):
                in_d[key] = in_d[key][0:n]#.detach()
            elif isinstance(in_d[key], list):
                for i in range(len(in_d[key])):
                    in_d[key][i] = in_d[key][i][0:n]#.detach()
                    
    return in_d

def dict_slice_se(in_d, keys, s, e):
    '''
    Apply type_as() of the dict-like.
    **** Every tensor must be in batch size (B x ...) ****
    :param in_d: a dict-like with {'key1': tensor, ...}
    :param target_d: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''
    for key in keys:
        if key in ['image_name', 'raw_image_path']:
            continue
        else:
            if th.is_tensor(in_d[key]):
                in_d[key] = in_d[key][s:e]#.detach()
            elif isinstance(in_d[key], list):
                for i in range(len(in_d[key])):
                    in_d[key][i] = in_d[key][i][s:e]#.detach()
                    
    return in_d