import os
import sys

import unireps
import numpy as np
import torch
import pickle
import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets.disable_caching()


if __name__ == "__main__":
    # Load command-line arguments
    hf_cache_dir = sys.argv[1]
    datasets_dir = sys.argv[2]
    outputs_dir = sys.argv[3]
    knn_dir = sys.argv[4]
    fig_dir = sys.argv[5]
    fig_cache_dir = sys.argv[6]
    
    unireps.set_hf_cache_directory(hf_cache_dir)
    unireps.set_datasets_directory(datasets_dir)
    unireps.set_outputs_directory(outputs_dir)

    # Setup directories
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if not os.path.exists(fig_cache_dir):
        os.makedirs(fig_cache_dir)

    ##### Big matrices #####

    def get_knn_path(model, dataset, use_chat_template=False):
        return os.path.join(knn_dir, unireps.get_dataset_name(model, dataset, use_chat_template=use_chat_template)) + '.parquet'

    def get_layer_knn(model, dataset, use_chat_template=False, k=10):
        ds = datasets.Dataset.from_parquet(get_knn_path(model, dataset, use_chat_template=use_chat_template))
        ds.set_format('torch')
        return ds['knn'][:,:,:k].permute(1,0,2)

    def generate_all_mknn(model_names, dataset_name):
        model_knns = {}
        for m in tqdm(model_names):
            model_knns[m] = get_layer_knn(m, dataset_name)

        model_model_mknn = {}
        for i in tqdm(range(len(model_names))):
            for j in range(i+1):
                m1 = model_names[i]
                m2 = model_names[j]
                knn_1 = model_knns[m1]
                knn_2 = model_knns[m2]
                mknn = unireps.mutual_knn(knn_1, knn_2)

                model_model_mknn[(m1,m2)] = mknn

        return model_model_mknn

    def generate_all_mknn_cached(path, model_names, dataset_name):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                mknns = pickle.load(f)
        else:
            mknns = generate_all_mknn(model_names, dataset_name)
            with open(path, 'wb') as f:
                pickle.dump(mknns, f)

        return mknns

    def get_from_all_mknn(mknns, m1, m2):
        if (m1,m2) in mknns:
            return mknns[(m1,m2)][1:,1:]
        else:
            return mknns[(m2,m1)][1:,1:].T
        
    def save_big_mat(path, mat_model_names, mknns):
        big_mat = torch.cat([torch.cat([get_from_all_mknn(mknns, m1, m2).T for m2 in mat_model_names]).T for m1 in mat_model_names])
        model_layers = np.array([get_from_all_mknn(mknns, m, mat_model_names[0]).shape[0] for m in mat_model_names])
        tick_positions = model_layers.cumsum()
        tick_positions = np.insert(tick_positions, 0, 0)
        tick_positions = (tick_positions[1:] + tick_positions[:-1])/2

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot()

        cax = ax.imshow(big_mat, vmin=0, vmax=1, interpolation="nearest", aspect='equal')
        # fig.colorbar(cax, fraction=0.02)

        ax.xaxis.set_tick_params(rotation=90)

        ax.xaxis.set_ticks(tick_positions, mat_model_names)
        ax.yaxis.set_ticks(tick_positions, mat_model_names)

        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(path, transparent=True, format='pdf')

    def make_big_mats(model_names, mat_model_names, dataset):
        mat_path = os.path.join(fig_dir, 'big_mat_{}.pdf'.format(dataset))
        if not os.path.exists(mat_path):
            mknn_path = os.path.join(fig_cache_dir, 'mknn_{}.pickle'.format(dataset))
            mknns = generate_all_mknn_cached(mknn_path, model_names, 'web_text')

            save_big_mat(mat_path, mat_model_names, mknns)

        mat_path = os.path.join(fig_dir, 'mega_mat_{}.pdf'.format(dataset))
        if not os.path.exists(mat_path):
            mknn_path = os.path.join(fig_cache_dir, 'mknn_{}.pickle'.format(dataset))
            mknns = generate_all_mknn_cached(mknn_path, model_names, 'web_text')

            save_big_mat(mat_path, model_names, mknns)

    model_names = [
        "openai-community/gpt2",
        "google/gemma-2b",
        "google/gemma-7b",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mixtral-8x7B-v0.1",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-11B",
        "tiiuae/falcon-mamba-7b"
    ]

    mat_model_names = [
        # "openai-community/gpt2",
        "google/gemma-2b",
        "google/gemma-7b",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        # "google/gemma-2-9b-it",
        "google/gemma-2-27b",
        "meta-llama/Meta-Llama-3.1-8B",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        # "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.1-70B",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mixtral-8x7B-v0.1",
        # "microsoft/Phi-3-mini-4k-instruct",
        # "microsoft/Phi-3-medium-4k-instruct",
        # "microsoft/Phi-3.5-mini-instruct",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-11B",
        # "tiiuae/falcon-mamba-7b"
    ]

    make_big_mats(model_names, mat_model_names, 'web_text')
    make_big_mats(model_names, mat_model_names, 'random_strings')
    make_big_mats(model_names, mat_model_names, 'book_translations_en')


    ##### Affinity plots #####

    def model_dataset_knn(model, dataset, use_chat_template=False, normalize=True, agg='last', k=10, n=None):
        ds = unireps.get_dataset(model, dataset, use_chat_template=use_chat_template)

        if n is not None:
            ds = ds.take(n)

        knn = unireps.embs_knn(unireps.dataset_embs(ds, layer=None, agg=agg, normalize=normalize), k=k)
        return knn

    def affinity_matrix_plot(output_name, model_1, model_2, dataset):
        print('Generating', output_name)
        output_path = os.path.join(fig_dir, output_name)
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
