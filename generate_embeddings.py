import sys
import os
import torch
import transformers
import datasets
import huggingface_hub


def embeddings_dataset(dataset, model, tokenizer):

    def compute_embeddings(example):
        tokens = tokenizer(example['text'], return_tensors='pt')
        input_ids = tokens['input_ids'].to(model.device)
        
        with torch.no_grad():
            model_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        
        # The hidden states are cast to float32 becauase it avoids some numerical issues, and everything
        # will have to be cast eventually because Arrow doesn't currently support types like bfloat16.
        layer_token_embeddings = torch.stack(model_out.hidden_states)[:,0].float()

        # Outputs are moved to the CPU to avoid memory problems on the GPU.
        layer_last_embeddings = layer_token_embeddings[:,-1].cpu()
        layer_mean_embeddings = layer_token_embeddings.mean(1).cpu()

        return {
            'layer_last_embeddings': layer_last_embeddings,
            'layer_mean_embeddings': layer_mean_embeddings
        }

    # TODO: This currently sets new_fingerprint because otherwise `map` appears to hash compute_embeddings which includes the entire model!
    # This does not use batching. In experiments, batching (somehow) slightly slowed it down across models and hardware!
    embeddings = dataset.map(compute_embeddings, new_fingerprint='fingerprint')
    embeddings.set_format('torch')
    
    return embeddings


if __name__ == "__main__":
     # Load command-line arguments
    model_list = sys.argv[1]
    cache_dir = sys.argv[2]
    dataset_list = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Authenticate for HF
    huggingface_hub.login(new_session=False)

    # Disable caching because we don't need it and it causes performance problems with big models
    datasets.disable_caching()

    for model_name in []:

        # Setup the scope for the model and tokenizer. They will only be loaded if they are needed
        tokenizer = None
        model = None

        for dataset_name in []:

            # TODO: Write this
            output_path = os.path.join(output_dir, '...')

            if os.path.isdir(output_path):
                continue

            # Load the model and the tokenizer if needed
            if tokenizer is None and model is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map='auto', cache_dir=cache_dir)
                model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto', cache_dir=cache_dir)

            # Load the dataset
            # trust_remote_code=False
            dataset = datasets.load_dataset(dataset_name, split='train', streaming=True, cache_dir=cache_dir)

            # Generate embeddings
            embeddings = embeddings_dataset(dataset, model, tokenizer)

            embeddings.save_to_disk(os.)