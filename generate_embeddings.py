import sys
import huggingface_hub
import datasets
import unireps

if __name__ == "__main__":
    # Load command-line arguments
    hf_cache_dir = sys.argv[1]
    datasets_dir = sys.argv[2]
    outputs_dir = sys.argv[3]
    
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
        # "web_text_caesar",
        "imdb",
        "random_strings",
        "book_translations_en",
        "book_translations_de",
        "common_words",
        "ifeval",
        "mmlu",
        "wikipedia"
    ]

    # Authenticate for HF
    huggingface_hub.login(new_session=False)

    # Disable caching because we don't need it
    datasets.disable_caching()

    print("Saving datasets...")
    unireps.save_datasets(datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)
    print("Finished saving datasets")

    unireps.generate_embeddings(
        model_names,
        dataset_names,
        chat_models,
        datasets_dir=datasets_dir,
        hf_cache_dir=hf_cache_dir,
        outputs_dir=outputs_dir,
        print_progress=True
    )
