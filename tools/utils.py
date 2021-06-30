import os
import json
import pickle 
import pandas as pd
import numpy as np
import cv2
import shutil

def read_csv(file_path):
    return pd.read_csv(file_path)
    
def avg_list(lyst):
    if lyst == []:
        return None
    else:
        return sum(lyst)/len(lyst)

def read_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def round_down(num, divisor):
    return num - (num%divisor)



def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)

def mkdir_replc(dirp):
    if os.path.exists(dirp):
        shutil.rmtree(dirp)
    os.makedirs(dirp)

def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def split_dict_ratio(data, ratio):
    data_list = list(data)
    list1, list2 = split_list_ratio(data_list, ratio)
    return dict(list1), dict(list2)


def split_list_ratio(data, ratio):
    rng = np.random.Generator(np.random.PCG64(2667))
    cutoff = round(len(data)*ratio)
    data.sort()
    rng.shuffle(data)
    return data[:cutoff], data[cutoff:]


def img_merge(imgs, mode, direction):
    """
    list_im: A list of images as numpy arrays
    mode: greyscale, RGB
    direction: vertical or horizontal stack
    Take a list of cv2 image objects and return one with them combined
    """
    if mode == "greyscale":
        if direction == "horizontal":
            imgs_comb = cv2.hconcat([i.byte().numpy() for i in imgs])
        elif direction == "vertical":
            imgs_comb = cv2.vconcat([i.byte().numpy() for i in imgs])
        else:
            raise Exception("Not implemented %s direction stacking" % (direction))
        return imgs_comb
    else:
        raise Exception("Not implemented %s mode" % (mode))


def model_fwd(model, frames, args):
    """
    Wrapper function to handle forward pass
    Transformer modesl need flat inputs and CNNs dont, this avoids clutter
    """
    if args.model == "transformer":
        b, in_no, w, h = frames.shape
        frames = frames.reshape(b, in_no, -1)
    out = model(frames)
    if args.model == "transformer":
        out = out.reshape(b, args.out_no, w, h)
    return out


