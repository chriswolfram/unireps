import sys
import os
import datasets
import torch
import huggingface_hub

# Basic IMDB

def prepare_imdb(cache_dir, output_dir):
    output_path = os.path.join(output_dir, 'imdb')
    if os.path.isdir(output_path):
        return
    
    # TODO: Remove unused columns?
    imdb = datasets.load_dataset('stanfordnlp/imdb', split='train[:4096]', cache_dir=cache_dir)
    imdb.save_to_disk(os.path.join(output_dir, 'imdb'))


# Caesar ciphered IMDB

def make_cipher(texts):
    chars = set()
    for t in texts:
        chars = chars.union(set(t))

    chars = list(chars)

    perm = torch.randperm(len(chars))
    cipher = {}
    for c, i in zip(chars, perm):
        cipher[c] = chars[i]

    return cipher

def apply_cipher(t, cipher):
    out = ''
    for c in t:
        out += cipher[c]
    return out

def prepare_imdb_caesar(cache_dir, output_dir):
    output_path = os.path.join(output_dir, 'imdb_caesar')
    if os.path.isdir(output_path):
        return
    
    # TODO: Remove unused columns?
    imdb = datasets.load_dataset('stanfordnlp/imdb', split='train[:4096]', cache_dir=cache_dir)
    cipher = make_cipher(imdb['text'])
    imdb_caesar = imdb.map(lambda x: {'text': apply_cipher(x['text'], cipher)})
    imdb_caesar.save_to_disk(os.path.join(output_dir, 'imdb_caesar'))


def prepare_random_strings(cache_dir, output_dir):
    output_path = os.path.join(output_dir, 'random')
    if os.path.isdir(output_path):
        return
    
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"

    random_strings_list = []
    for _ in range(4096):
        random_strings_list.append({'text': ''.join([chars[i] for i in torch.randint(len(chars), (100,))])})

    random_strings = datasets.Dataset.from_list(random_strings_list)
    random_strings.save_to_disk(os.path.join(output_dir, 'random_strings'))

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

    print("Preparing Caesar IMDB")
    prepare_imdb_caesar(cache_dir, datasets_dir)

    print("Preparing random strings")
    prepare_random_strings(cache_dir, datasets_dir)

    # ds_streaming = datasets.load_dataset('stanfordnlp/imdb', split='train', streaming=True, cache_dir=cache_dir, trust_remote_code=False)

    # openwebtext
    # the pile
    # google/wit

    # shuffling?