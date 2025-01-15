import os
import transformers
import datasets
import torch
from . import paths
from . import representations

def get_dataset_name(model_name, dataset_name, use_chat_template=False):
    model_label = model_name.replace('/', '__')

    if use_chat_template:
        model_label += "---chat"

    return '{}---{}'.format(model_label, dataset_name)

def get_dataset_path(model_name, dataset_name, use_chat_template=False, outputs_dir=None):    
    if outputs_dir is None:
        outputs_dir = paths.outputs_directory

    return os.path.join(outputs_dir, get_dataset_name(model_name, dataset_name, use_chat_template))


def get_dataset(model_name, dataset_name, use_chat_template=False, outputs_dir=None):
    return datasets.load_from_disk(get_dataset_path(model_name, dataset_name, use_chat_template, outputs_dir))


def dataset_embs(ds, layer=None, agg='last', normalize=True):
    assert agg == 'last' or agg == 'mean'
    if agg == 'last':
        key = 'layer_last_embeddings'
    elif agg == 'mean':
        key = 'layer_mean_embeddings'

    if layer is None:
        embs = torch.stack([e[key] for e in ds]).permute(1,0,2)
    else:
        embs = torch.stack([e[key][layer] for e in ds])

    if normalize:
        embs = torch.nn.functional.normalize(embs, dim=-1)

    return embs


def generate_embeddings(model_names, dataset_names, chat_models=[], datasets_dir=None, hf_cache_dir=None, outputs_dir=None, print_progress=True):
    if datasets_dir is None:
        datasets_dir = paths.datasets_directory

    if hf_cache_dir is None:
        hf_cache_dir = paths.hf_cache_directory
    
    if outputs_dir is None:
        outputs_dir = paths.outputs_directory


    for model_name in model_names:

        chat_options = [False]
        if model_name in chat_models:
            chat_options.append(True)

        # Setup the scope for the model and tokenizer. They will only be loaded if they are needed
        tokenizer = None
        model = None

        for dataset_name in dataset_names:
            for use_chat_template in chat_options:

                torch.cuda.empty_cache()

                output_path = dataset_path(model_name, dataset_name, use_chat_template, outputs_dir)

                if print_progress:
                    if use_chat_template:
                        print('Generating {} on {} with chat template'.format(model_name, dataset_name))
                    else:
                        print('Generating {} on {}'.format(model_name, dataset_name))

                if os.path.isdir(output_path):
                    if print_progress:
                        print('Output already exists (skipping)')
                    continue

                # Load the model and the tokenizer if needed
                if tokenizer is None and model is None:
                    if print_progress:
                        print('Loading model:\t{}'.format(model_name))
                        
                    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map='auto', cache_dir=hf_cache_dir)
                    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto', cache_dir=hf_cache_dir)

                # Load the dataset
                dataset = datasets.load_from_disk(os.path.join(datasets_dir, dataset_name))

                # Generate embeddings
                embeddings = representations.embed_dataset(dataset, model, tokenizer, use_chat_template)

                embeddings.save_to_disk(output_path)
    