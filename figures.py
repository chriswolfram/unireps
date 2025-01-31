import os
import sys

import unireps
import numpy as np
import scipy
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

    def generate_all_mknn(model_names, dataset_name_1, dataset_name_2, k=10):
        print("Generating mknn for:", dataset_name_1, dataset_name_2)
        model_knns_1 = {}
        for m in tqdm(model_names):
            model_knns_1[m] = get_layer_knn(m, dataset_name_1, k=k)

        model_knns_2 = {}
        for m in tqdm(model_names):
            model_knns_2[m] = get_layer_knn(m, dataset_name_2, k=k)

        model_model_mknn = {}
        for i in tqdm(range(len(model_names))):
            for j in range(i+1):
                m1 = model_names[i]
                m2 = model_names[j]
                knn_1 = model_knns_1[m1]
                knn_2 = model_knns_2[m2]
                mknn = unireps.mutual_knn(knn_1, knn_2)

                model_model_mknn[(m1,m2)] = mknn

        return model_model_mknn

    def generate_all_mknn_cached(path, model_names, dataset_name_1, dataset_name_2, k=10):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                mknns = pickle.load(f)
        else:
            mknns = generate_all_mknn(model_names, dataset_name_1, dataset_name_2, k=k)
            with open(path, 'wb') as f:
                pickle.dump(mknns, f)

        return mknns

    def get_from_all_mknn(mknns, m1, m2):
        if (m1,m2) in mknns:
            return mknns[(m1,m2)][1:,1:]
        else:
            return mknns[(m2,m1)][1:,1:].T

    def make_big_mats(model_names, mat_model_names, dataset_1, dataset_2, cache_suffix='', k=10):
        mat_path = os.path.join(fig_dir, 'big_mat_{}_{}{}.pdf'.format(dataset_1, dataset_2, cache_suffix))
        print("Making big mat:", mat_path)
        if not os.path.exists(mat_path):
            mknn_path = os.path.join(fig_cache_dir, 'mknn_{}_{}{}.pickle'.format(dataset_1, dataset_2, cache_suffix))
            mknns = generate_all_mknn_cached(mknn_path, model_names, dataset_1, dataset_2, k=k)

            unireps.big_mat_plot(mknns, mat_model_names, tick_spacing=15, rotate_model_names=True, figsize=(10,10))
            plt.tight_layout()
            plt.savefig(mat_path, transparent=True, format='pdf')

        mat_path = os.path.join(fig_dir, 'mega_mat_{}_{}{}.pdf'.format(dataset_1, dataset_2, cache_suffix))
        if not os.path.exists(mat_path):
            mknn_path = os.path.join(fig_cache_dir, 'mknn_{}_{}{}.pickle'.format(dataset_1, dataset_2, cache_suffix))
            mknns = generate_all_mknn_cached(mknn_path, model_names, dataset_1, dataset_2, k=k)

            unireps.big_mat_plot(mknns, model_names, tick_spacing=10000, rotate_model_names=True, figsize=(15,15))
            plt.tight_layout()
            plt.savefig(mat_path, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)

    model_names = [
        # "openai-community/gpt2",
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
    model_names.reverse()

    mat_model_names = [
        # "openai-community/gpt2",
        # "google/gemma-2b",
        # "google/gemma-7b",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        # "google/gemma-2-9b-it",
        "google/gemma-2-27b",
        "meta-llama/Meta-Llama-3.1-8B",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.1-70B",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        # "mistralai/Mixtral-8x7B-v0.1",
        # "microsoft/Phi-3-mini-4k-instruct",
        # "microsoft/Phi-3-medium-4k-instruct",
        # "microsoft/Phi-3.5-mini-instruct",
        "tiiuae/falcon-40b",
        # "tiiuae/falcon-11B",
        "tiiuae/falcon-mamba-7b"
    ]
    mat_model_names.reverse()

    make_big_mats(model_names, mat_model_names, 'web_text', 'web_text')
    make_big_mats(model_names, mat_model_names, 'random_strings', 'random_strings')
    make_big_mats(model_names, mat_model_names, 'book_translations_en', 'book_translations_en')
    make_big_mats(model_names, mat_model_names, 'book_translations_de', 'book_translations_de')
    make_big_mats(model_names, mat_model_names, 'imdb', 'imdb')
    make_big_mats(model_names, mat_model_names, 'ifeval', 'ifeval')
    make_big_mats(model_names, mat_model_names, 'mmlu', 'mmlu')

    # make_big_mats(model_names, mat_model_names, 'book_translations_en', 'book_translations_de')
    # make_big_mats(model_names, mat_model_names, 'book_translations_de', 'book_translations_en')

    make_big_mats(model_names, mat_model_names, 'web_text', 'web_text', k=1, cache_suffix='_k1')
    # make_big_mats(model_names, mat_model_names, 'web_text', 'web_text', k=100, cache_suffix='_k100')


    ##### Unit square plots #####

    def unit_square_mat(mknns, bins=40):
        x = []
        y = []
        z = []
        for _, mknn_raw in list(mknns.items()):
            for mknn in [mknn_raw, mknn_raw.T]:
                coords = torch.stack(torch.meshgrid(torch.linspace(0,1,mknn.shape[0]), torch.linspace(0,1,mknn.shape[1]), indexing='ij'), dim=-1).flatten(end_dim=1)
                x.append(coords[:,0])
                y.append(coords[:,1])
                z.append(mknn.flatten())

        x = torch.cat(x)
        y = torch.cat(y)
        z = torch.cat(z)


        stat, _, _, _ = scipy.stats.binned_statistic_2d(x, y, z, statistic='mean', bins=bins)

        return stat

    def clean_up_mknns(mknns):
        out = mknns
        for k in list(out.keys()):
            if "openai-community/gpt2" in k or k[0] == k[1]:
                del out[k]

        out = {k: v[1:,1:] for k,v in out.items()}
        return out

    output_path = os.path.join(fig_dir, 'triple_unit_plot.pdf')
    if not os.path.exists(output_path):    
        mknns_web_text = clean_up_mknns(generate_all_mknn_cached(os.path.join(fig_cache_dir, 'mknn_web_text_web_text.pickle'), model_names, 'web_text', 'web_text'))
        mknns_random_strings = clean_up_mknns(generate_all_mknn_cached(os.path.join(fig_cache_dir, 'mknn_random_strings_random_strings.pickle'), model_names, 'random_strings', 'random_strings'))
        mknns_ifeval = clean_up_mknns(generate_all_mknn_cached(os.path.join(fig_cache_dir, 'mknn_ifeval_ifeval.pickle'), model_names, 'ifeval', 'ifeval'))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        vmax = 0.65

        axs[0].imshow(unit_square_mat(mknns_web_text, bins=40).T, origin='lower', extent=[0,1,0,1], vmin=0, vmax=vmax, cmap='inferno')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('proportional depth in model 1')
        axs[0].set_ylabel('proportional depth in model 2')

        axs[1].imshow(unit_square_mat(mknns_random_strings, bins=40).T, origin='lower', extent=[0,1,0,1], vmin=0, vmax=vmax, cmap='inferno')
        axs[1].set_aspect('equal')
        axs[1].set_xlabel('proportional depth in model 1')
        # axs[1].set_ylabel('proportional depth in model 2')

        im = axs[2].imshow(unit_square_mat(mknns_ifeval, bins=40).T, origin='lower', extent=[0,1,0,1], vmin=0, vmax=vmax, cmap='inferno')
        axs[2].set_aspect('equal')
        axs[2].set_xlabel('proportional depth in model 1')
        # axs[2].set_ylabel('proportional depth in model 2')

        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label='mean similarity')

        # plt.tight_layout()
        plt.savefig(output_path, transparent=True, dpi=200, format='pdf', bbox_inches='tight', pad_inches=0)


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
            plt.savefig(output_path, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)


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
