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
    
    imdb = datasets.load_dataset('stanfordnlp/imdb', split='test', cache_dir=cache_dir).shuffle(seed=1234).take(4096)
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
    
    imdb = datasets.load_from_disk(os.path.join(output_dir, 'imdb'))
    cipher = make_cipher(imdb['text'])
    imdb_caesar = imdb.map(lambda x: {'text': apply_cipher(x['text'], cipher)})
    imdb_caesar.save_to_disk(os.path.join(output_dir, 'imdb_caesar'))


# Random strings

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


# Web text

def prepare_web_text(cache_dir, output_dir):
    output_path = os.path.join(output_dir, 'web_text')
    if os.path.isdir(output_path):
        return
    
    owt = datasets.load_dataset('Skylion007/openwebtext', split='train', cache_dir=cache_dir).shuffle().to_iterable_dataset().filter(lambda x: len(x['text']) < 10000).take(4096)
    owt.save_to_disk(os.path.join(output_dir, 'web_text'))

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

    print("Preparing web text")
    prepare_web_text(cache_dir, datasets_dir)

    # openwebtext
    # the pile
    # google/wit
    # tiiuae/falcon-refinedweb