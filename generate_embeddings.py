import sys
import os
import json
import torch
import transformers
import datasets
import huggingface_hub


def embeddings_dataset(dataset, model, tokenizer, use_chat_template=False):

    def compute_embeddings(example):
        if use_chat_template:
            messages = [{'role': 'user', 'content': example['text']}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, truncation=True, return_tensors='pt')
            input_ids = input_ids.to(model.device)
        else:
            tokens = tokenizer(example['text'], truncation=True, return_tensors='pt')
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


def output_name(model_name, dataset_name, use_chat_template):
    model_label = model_name.replace('/', '__')

    if use_chat_template:
        model_label += "---chat"

    return '{}---{}'.format(model_label, dataset_name)


if __name__ == "__main__":
     # Load command-line arguments
    run_path = sys.argv[1]
    cache_dir = sys.argv[2]
    dataset_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Authenticate for HF (just in case)
    huggingface_hub.login(new_session=False)

    # Disable caching because we don't need it and it causes performance problems with big models
    datasets.disable_caching()

    # Get models and datasets
    with open(run_path) as f:
        run_json = json.load(f)

    models_infos = run_json['models']
    dataset_infos = run_json['datasets']

    print('Models:')
    for m in models_infos:
        print("- {} (chat: {})".format(m['name'], m['chat']))

    print('Datasets:')
    for d in dataset_infos:
        print("- {}".format(d['name']))

    for model_info in models_infos:
        model_name = model_info['name']
        model_use_chat_template = model_info['chat']

        chat_options = [False]
        if model_use_chat_template:
            chat_options.append(True)

        # Setup the scope for the model and tokenizer. They will only be loaded if they are needed
        tokenizer = None
        model = None

        for dataset_info in dataset_infos:
            dataset_name = dataset_info['name']

            for use_chat_template in chat_options:

                torch.cuda.empty_cache()

                output_path = os.path.join(output_dir, output_name(model_name, dataset_name, use_chat_template))
                print('Generating:\t', output_path)

                if os.path.isdir(output_path):
                    print('Skipping:\t', output_path)
                    continue

                # Load the model and the tokenizer if needed
                if tokenizer is None and model is None:
                    print('Loading model:\t', model_name)
                    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map='auto', cache_dir=cache_dir)
                    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto', cache_dir=cache_dir)

                # Load the dataset
                dataset = datasets.load_from_disk(os.path.join(dataset_dir, dataset_name))

                # Generate embeddings
                embeddings = embeddings_dataset(dataset, model, tokenizer, use_chat_template)

                embeddings.save_to_disk(output_path)