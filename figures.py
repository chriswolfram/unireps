import os
import sys

import unireps
import torch
import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

figure_directory = os.path.abspath('./figures')
if not os.path.exists(figure_directory):
    os.makedirs(figure_directory)

project_dir = '/net/scratch2/chriswolfram/unireps'
unireps.set_hf_cache_directory(os.path.join(project_dir, 'hf_cache'))
unireps.set_datasets_directory(os.path.join(project_dir, 'datasets'))
unireps.set_outputs_directory(os.path.join(project_dir, 'outputs'))

datasets.disable_caching()


def model_dataset_knn(model, dataset, use_chat_template=False, normalize=True, agg='last', k=10, n=None):
    ds = unireps.get_dataset(model, dataset, use_chat_template=use_chat_template)

    if n is not None:
        ds = ds.take(n)

    knn = unireps.embs_knn(unireps.dataset_embs(ds, layer=None, agg=agg, normalize=normalize), k=k)
    return knn


##### Figure 1 #####

output_name = 'web_text_symmetric_affinity.pdf'
output_path = os.path.join(figure_directory, output_name)
if not os.path.exists(output_path):
    knn_1 = model_dataset_knn("meta-llama/Meta-Llama-3.1-8B", "web_text")
    mknn = unireps.mutual_knn(knn_1, knn_1)
    unireps.layer_by_layer_plot(mknn[1:,1:], x_label="layer of meta-llama/Meta-Llama-3.1-8B", y_label="layer of meta-llama/Meta-Llama-3.1-8B")
    plt.tight_layout()
    plt.savefig(output_path, transparent=True, format='pdf')


output_name = 'web_text_affinity.pdf'
output_path = os.path.join(figure_directory, output_name)
if not os.path.exists(output_path):
    knn_1 = model_dataset_knn("meta-llama/Meta-Llama-3.1-8B", "web_text")
    knn_2 = model_dataset_knn("google/gemma-2-9b", "web_text")
    mknn = unireps.mutual_knn(knn_1, knn_2)
    unireps.layer_by_layer_plot(mknn[1:,1:], x_label="layer of google/gemma-2-9b", y_label="layer of meta-llama/Meta-Llama-3.1-8B")
    plt.tight_layout()
    plt.savefig(output_path, transparent=True, format='pdf')


output_name = 'random_strings_affinity.pdf'
output_path = os.path.join(figure_directory, output_name)
if not os.path.exists(output_path):
    knn_1 = model_dataset_knn("meta-llama/Meta-Llama-3.1-8B", "random_strings")
    knn_2 = model_dataset_knn("google/gemma-2-9b", "random_strings")
    mknn = unireps.mutual_knn(knn_1, knn_2)
    unireps.layer_by_layer_plot(mknn[1:,1:], x_label="layer of google/gemma-2-9b", y_label="layer of meta-llama/Meta-Llama-3.1-8B")
    plt.tight_layout()
    plt.savefig(output_path, transparent=True, format='pdf')