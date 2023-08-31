import blobfile as bf

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

def _list_npy_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_npy_files_recursively(full_path))
    return results

def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mp4"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_video_files_recursively(full_path))
    return results

def list_path_to_dict(list_path, force_type=None):
    """
    Convert list of image path into dictionary with the {<image_name> : <image_path>}
    e.g. {'0.jpg' : '/data/mint/ffhq_256/0.jpg'}
    :param list_path: list of image path
    """
    dict_path = {}
    for tmp in list_path:
        key = tmp.split('/')[-1]
        if force_type is not None:
            dict_path[key.split('.')[0] + force_type] = tmp
        else:
            dict_path[key] = tmp
    return dict_path
        
def search_index_from_listpath(list_path, search):
    """
    Return the index form image name from search list
    :param list_path: list of image path
    :param search: list of image name to search in list_path
    """
    img_idx = [None, None]
    list_path = [path.split('/')[-1] for path in list_path]
    try:
        img_idx = [list_path.index(s) for s in search]
    except ValueError:
        raise(f"[#] {search} cannot be found in image path list")
    return img_idx
