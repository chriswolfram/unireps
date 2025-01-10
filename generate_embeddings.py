import sys
import os
import torch
import transformers
import datasets
import huggingface_hub


force_no_chat_models = [
    'tiiuae/falcon-11B',
    'tiiuae/falcon-40b'
]


def embeddings_dataset(dataset, model, tokenizer, force_no_chat=False):

    def compute_embeddings(example):
        if tokenizer.chat_template is None or force_no_chat:
            tokens = tokenizer(example['text'], truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(model.device)
        else:
            messages = [{'role': 'user', 'content': example['text']}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, truncation=True, return_tensors='pt')
            input_ids = input_ids.to(model.device)
        
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


def output_name(model_name, dataset_name):
    return '{}---{}'.format(model_name.replace('/', '__'), dataset_name)


if __name__ == "__main__":
     # Load command-line arguments
    model_list = sys.argv[1]
    cache_dir = sys.argv[2]
    dataset_list = sys.argv[3]
    dataset_dir = sys.argv[4]
    output_dir = sys.argv[5]
    
    # Authenticate for HF (just in case)
    huggingface_hub.login(new_session=False)

    # Disable caching because we don't need it and it causes performance problems with big models
    datasets.disable_caching()

    # Get models
    model_names = []
    with open(model_list) as f:
        for model_name_nl in f:
            model_name = model_name_nl.rstrip()
            model_names.append(model_name)

    # Get datasets
    dataset_names = []
    with open(dataset_list) as f:
        for dataset_name_nl in f:
            dataset_name = dataset_name_nl.rstrip()
            dataset_names.append(dataset_name)

    print('Models:')
    for m in model_names:
        print("- {}".format(m))

    print('Datasets:')
    for d in dataset_names:
        print("- {}".format(d))

    for model_name in model_names:

        # Setup the scope for the model and tokenizer. They will only be loaded if they are needed
        tokenizer = None
        model = None
        force_no_chat = False

        for dataset_name in dataset_names:

            torch.cuda.empty_cache()

            output_path = os.path.join(output_dir, output_name(model_name, dataset_name))
            print('Generating:\t', output_path)

            if os.path.isdir(output_path):
                print('Skipping:\t', output_path)
                continue

            # Load the model and the tokenizer if needed
            if tokenizer is None and model is None:
                print('Loading model:\t', model_name)
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map='auto', cache_dir=cache_dir)
                model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto', cache_dir=cache_dir)

                print('Chat model:\t', tokenizer.chat_template is not None)
                force_no_chat = False
                if model_name in force_no_chat_models:
                    force_no_chat = True
                    print('Forcing no chat:\t', force_no_chat)

            # Load the dataset
            dataset = datasets.load_from_disk(os.path.join(dataset_dir, dataset_name))

            # Generate embeddings
            embeddings = embeddings_dataset(dataset, model, tokenizer, force_no_chat)

            embeddings.save_to_disk(output_path)