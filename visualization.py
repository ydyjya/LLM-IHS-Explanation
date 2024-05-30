import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from scipy.spatial import distance
from scipy.special import softmax
from scipy.stats import entropy
from collections import Counter
from emotion_token import neg_list, pos_list, neutral_list
from mpl_toolkits.axes_grid1 import make_axes_locatable

def topk_intermediate_confidence_heatmap(forward_info, topk=5, layer_nums=32, left=0, right=33, model_name="", dataset_size=100):
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
    filtered_res_top_k = {key: res_top_k[key] for key in selected_keys}

    keys = list(filtered_res_top_k.keys())
    words = [item[0] for sublist in filtered_res_top_k.values() for item in sublist]
    counts = [item[1] for sublist in filtered_res_top_k.values() for item in sublist]

    heatmap_data = np.array(counts).reshape(len(keys), topk).T
    word_labels = np.array(words).reshape(len(keys), topk).T

    fig, ax = plt.subplots(figsize=((right-left)*2, (right-left)))
    cax = ax.matshow(heatmap_data, cmap='viridis', vmin=0, vmax=dataset_size)
    divider = make_axes_locatable(ax)
    cax_colorbar = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(cax, cax=cax_colorbar)

    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f'L{key}' for key in keys])
    ax.set_yticks(np.arange(topk))
    ax.set_yticklabels([f"{_i}" for _i in range(topk)])
    for i in range(topk):
        for j in range(len(keys)):
            if word_labels[i, j].lower() in neg_list:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='Red', fontsize=15,fontdict={'weight': 'bold'}, rotation=45)
            elif word_labels[i, j].lower() in neutral_list:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='purple', fontsize=15,fontdict={'weight': 'bold'})
            elif word_labels[i, j].lower() in pos_list:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='Green', fontsize=15,fontdict={'weight': 'bold'}, rotation=315)
            else:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='black', fontsize=15)

    ax.set_title(f'Top {topk} from Intermediate Hidden States \n (Layer {left}-{right})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Rank')

    plt.tight_layout()
    plt.savefig(f"./vis/{model_name}_{left}_{right}.png")
    plt.show()


def accuracy_line(rep_dict, model_name):
    fig, axs = plt.subplots(1, len(rep_dict), figsize=(20, 20))
    if len(rep_dict) == 1:
        axs = [axs]
    idx = 0
    for classifier, layers_rep in rep_dict.items():
        acc_list = []
        for layer, rep in layers_rep.items():
            acc_list.append(rep['accuracy'])
        x_range = [i for i in range(-1, len(acc_list) - 1)]
        axs[idx].plot(x_range, acc_list, label=f"{model_name}_{classifier}")
        axs[idx].set_title(f'{model_name}_{classifier}')
        axs[idx].set(xlabel='Layer', ylabel='Weak Classification Accuracy')
        axs[idx].tick_params(axis='x', rotation=15)
        axs[idx].legend()
        idx += 1

    plt.tight_layout(rect=[0, 0.6, 0.85, 1])
    plt.savefig(f"./vis/acc_{model_name}.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()