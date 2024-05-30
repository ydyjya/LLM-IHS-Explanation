import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_layer(forward_info, layer):
    new_forward_info = {}
    for k, v in forward_info.items():
        new_forward_info[k] = {"hidden_states": v["hidden_states"][layer], "top-value_pair": v["top-value_pair"][layer], "label":v["label"]}
    return new_forward_info
