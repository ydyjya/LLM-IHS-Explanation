import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from scipy.spatial import distance
from scipy.special import softmax
from scipy.stats import entropy
from collections import Counter


def topk_intermediate_confidence_heatmap(forward_info, topk=5, layer_nums=32, left=0, right=33, model_name="", dataset_size=500):
    top_k_kv = {_: {} for _ in range(0, layer_nums + 1)}
    for k, v in forward_info.items():
        for layer_idx, tv_pair_list in enumerate(v["top-value_pair"]):
            for top_k_pair in tv_pair_list:
                if top_k_pair[0] in top_k_kv[layer_idx]:
                    top_k_kv[layer_idx][top_k_pair[0]] += 1
                else:
                    top_k_kv[layer_idx][top_k_pair[0]] = 1
    res_top_k = {}
    for k, v in top_k_kv.items():
        counter = Counter(v)
        top_k_per_layer = counter.most_common(topk)
        res_top_k[k] = top_k_per_layer

    selected_keys = list(range(left, right))
    filtered_res_top_k = {key: res_top_k[key + 1] for key in selected_keys}

    keys = list(filtered_res_top_k.keys())
    words = [item[0] for sublist in filtered_res_top_k.values() for item in sublist]
    counts = [item[1] for sublist in filtered_res_top_k.values() for item in sublist]

    heatmap_data = np.array(counts).reshape(len(keys), topk).T
    word_labels = np.array(words).reshape(len(keys), topk).T

    fig, ax = plt.subplots(figsize=(20, 10))
    cax = ax.matshow(heatmap_data, cmap='viridis', vmin=0, vmax=dataset_size)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f'Layer {key}' for key in keys])
    ax.set_yticks(np.arange(topk))
    ax.set_yticklabels([f"{_i}" for _i in range(topk)])
    for i in range(topk):
        for j in range(len(keys)):
            if word_labels[i, j] in ["certainly", "commendiate", "congr", "hello", "Hello", "hi", "welcome", "welcome",
                                     "glad", "pleasure", "thank", "Thank", "delight", "gre", "Hi", "great", "Great"]:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='white', fontsize=15,
                        fontdict={'weight': 'bold'}, rotation=45)
            else:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='black', fontsize=15)

    fig.colorbar(cax)

    ax.set_title(f'Top {topk} from Intermediate Hidden States \n (Layer {left}-{right})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Rank')

    plt.tight_layout()
    plt.savefig(f"./vis/{model_name}_{left}_{right}.png")
    plt.show()


def accuracy(rep):
    pass