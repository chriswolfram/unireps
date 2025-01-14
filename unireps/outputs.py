import os
import transformers
import datasets
from . import constants
from . import representations

def embedding_output_name(model_name, dataset_name, use_chat_template=False):
    """
    Generate a standardized output name based on the model name and dataset name.

    Args:
        model_name (str): The name of the model, which may contain '/' characters.
        dataset_name (str): The name of the dataset.
        use_chat_template (bool, optional): If True, append "---chat" to the model label. Defaults to False.

    Returns:
        str: A standardized output name in the format 'model_label---dataset_name'.
    """

    model_label = model_name.replace('/', '__')

    if use_chat_template:
        model_label += "---chat"

    return '{}---{}'.format(model_label, dataset_name)

def embedding_output_path(model_name, dataset_name, use_chat_template=False, outputs_dir=None):
    """
    Generate the full output path for a given model and dataset.

    Args:
        model_name (str): The name of the model, which may contain '/' characters.
        dataset_name (str): The name of the dataset.
        use_chat_template (bool, optional): If True, append "---chat" to the model label. Defaults to False.
        outputs_dir (str, optional): The directory where outputs are stored. If None, defaults to 'outputs_directory'.

    Returns:
        str: The full path to the output file.
    """
    
    if outputs_dir is None:
        print(constants.outputs_directory)
        outputs_dir = constants.outputs_directory

    return os.path.join(outputs_dir, embedding_output_name(model_name, dataset_name, use_chat_template))


def generate_embeddings(model_names, dataset_names, chat_models=[], datasets_dir=None, hf_cache_dir=None, outputs_dir=None, print_progress=True):
    if datasets_dir is None:
        datasets_dir = constants.datasets_directory

    if hf_cache_dir is None:
        hf_cache_dir = constants.hf_cache_directory
    
    if outputs_dir is None:
        outputs_dir = constants.outputs_directory


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

                output_path = embedding_output_path(model_name, dataset_name, use_chat_template, outputs_dir)

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
    