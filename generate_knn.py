import sys
import os
import torch
import datasets
import unireps


def knn_mat(emb_mat, k=None):
    gram = emb_mat @ emb_mat.T
    gram.fill_diagonal_(-torch.inf)

    if k is None:
        k = gram.shape[0]-1

    nn = gram.argsort(descending=True)[:,:k]
    return nn


if __name__ == "__main__":
    # Load command-line arguments
    hf_cache_dir = sys.argv[1]
    datasets_dir = sys.argv[2]
    outputs_dir = sys.argv[3]
    knn_dir = sys.argv[4]
    
    unireps.set_hf_cache_directory(hf_cache_dir)
    unireps.set_datasets_directory(datasets_dir)
    unireps.set_outputs_directory(outputs_dir)
    
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
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mixtral-8x7B-v0.1",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-11B",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "tiiuae/falcon-mamba-7b"
    ]

    chat_models = [
        "google/gemma-2-9b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct"
    ]

    dataset_names = [
        "web_text",
        "web_text_caesar",
        "imdb",
        "random_strings",
        "book_translations_en",
        "book_translations_de",
        "common_words",
        "ifeval",
        "mmlu",
        "wikipedia"
    ]

    for model_name in model_names:
        chat_options = [False]
        if model_name in chat_models:
            chat_options.append(True)
                
        for use_chat_template in chat_options:
            for dataset_name in dataset_names:
                print('Generating', unireps.get_dataset_name(model_name, dataset_name, use_chat_template))
                output_path = os.path.join(knn_dir, unireps.get_dataset_name(model_name, dataset_name, use_chat_template)) + '.parquet'
                
                if os.path.exists(output_path):
                    print('Output exists. Skipping...')
                    continue
                
                if not os.path.isdir(unireps.get_dataset_path(model_name, dataset_name, use_chat_template)):
                    print('Source does not exist. Skipping...')
                    continue
                
                ds = unireps.get_dataset(model_name, dataset_name, use_chat_template)
                embs = unireps.dataset_embs(ds, normalize=True)
                knn = torch.stack([knn_mat(e) for e in embs]).permute(1,0,2).to(torch.int16)
                knn_ds = datasets.Dataset.from_dict({'text': ds['text'], 'knn': knn, 'at_max_length': ds['at_max_length']})
                knn_ds.to_parquet(output_path)