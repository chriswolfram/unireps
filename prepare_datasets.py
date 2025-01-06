import sys
import os
import datasets
import huggingface_hub

def prepare_imdb(cache_dir, output_dir):
    output_path = os.path.join(output_dir, 'imdb')
    if os.path.isdir(output_path):
        return
    
    # TODO: Remove unused columns?
    imdb = datasets.load_dataset('stanfordnlp/imdb', split='train[:4096]', cache_dir=cache_dir)
    imdb.save_to_disk(os.path.join(output_dir, 'imdb'))

if __name__ == "__main__":
     # Load command-line arguments
    cache_dir = sys.argv[1]
    datasets_dir = sys.argv[2]

    # Create output directory if needed
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    
    # Authenticate for HF
    huggingface_hub.login(new_session=False)

    # Disable caching because we don't need it
    datasets.disable_caching()

    print("Preparing IMDB")
    prepare_imdb(cache_dir, datasets_dir)

    # ds_streaming = datasets.load_dataset('stanfordnlp/imdb', split='train', streaming=True, cache_dir=cache_dir, trust_remote_code=False)

    # openwebtext
    # the pile
    # shuffling?