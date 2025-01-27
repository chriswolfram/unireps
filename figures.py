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

def affinity_matrix_plot(output_name, model_1, model_2, dataset):
    print('Generating', output_name)
    output_path = os.path.join(figure_directory, output_name)
    if not os.path.exists(output_path):
        knn_1 = model_dataset_knn(model_2, dataset)
        knn_2 = model_dataset_knn(model_1, dataset)
        mknn = unireps.mutual_knn(knn_1, knn_2)
        unireps.layer_by_layer_plot(mknn[1:,1:], x_label='layer of ' + model_1, y_label='layer of ' + model_2)
        plt.tight_layout()
        plt.savefig(output_path, transparent=True, format='pdf')


##### Figure 1 #####

affinity_matrix_plot('web_text_affinity.pdf', "google/gemma-2-9b", "meta-llama/Meta-Llama-3.1-8B", 'web_text')
affinity_matrix_plot('random_strings_affinity.pdf', "google/gemma-2-9b", "meta-llama/Meta-Llama-3.1-8B", 'random_strings')

##### Figure 2 #####

affinity_matrix_plot('web_text_instruction_tuning.pdf', "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B", 'web_text')
affinity_matrix_plot('ifeval_instruction_tuning.pdf', "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B", 'ifeval')

affinity_matrix_plot('web_text_gemma_instruction_tuning.pdf', "google/gemma-2-9b-it", "google/gemma-2-9b", 'web_text')
affinity_matrix_plot('ifeval_gemma_instruction_tuning.pdf', "google/gemma-2-9b-it", "google/gemma-2-9b", 'ifeval')

affinity_matrix_plot('random_strings_instruction_tuning.pdf', "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B", 'random_strings')

##### Appendix figures #####

### Zoo on web text
affinity_matrix_plot('web_text_llama_mistral_affinity.pdf', "meta-llama/Llama-3.1-70B", "mistralai/Mistral-7B-v0.3", 'web_text')
affinity_matrix_plot('web_text_falcon_gemma_affinity.pdf', "tiiuae/falcon-40b", "google/gemma-2-27b", 'web_text')
affinity_matrix_plot('web_text_mistral_gemma_affinity.pdf', "mistralai/Mistral-7B-v0.3", "google/gemma-2-2b", 'web_text')

### Zoo on random strings
affinity_matrix_plot('random_strings_llama_mistral_affinity.pdf', "meta-llama/Llama-3.1-70B", "mistralai/Mistral-7B-v0.3", 'random_strings')
affinity_matrix_plot('random_strings_falcon_gemma_affinity.pdf', "tiiuae/falcon-40b", "google/gemma-2-27b", 'random_strings')
affinity_matrix_plot('random_strings_mistral_gemma_affinity.pdf', "mistralai/Mistral-7B-v0.3", "google/gemma-2-2b", 'random_strings')

##### Other #####

affinity_matrix_plot('web_text_symmetric_affinity.pdf', "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B", 'web_text')
