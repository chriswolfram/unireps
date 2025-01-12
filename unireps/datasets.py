import os
import datasets
import torch
from .constants import *

def _get_dirs(datasets_dir, hf_cache_dir):
    if datasets_dir is None:
        datasets_dir = datasets_directory

    if hf_cache_dir is None:
        hf_cache_dir = hf_cache_directory

    return datasets_dir, hf_cache_dir

# Caesar ciphered datasets

def _make_cipher(texts, seed=1234):
    chars = set()
    for t in texts:
        chars = chars.union(set(t))

    chars = list(chars)

    torch.manual_seed(seed)
    perm = torch.randperm(len(chars))
    cipher = {}
    for c, i in zip(chars, perm):
        cipher[c] = chars[i]

    return cipher

def _apply_cipher(t, cipher):
    out = ''
    for c in t:
        out += cipher[c]
    return out

def save_caesar(input_dataset_name, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    output_path = os.path.join(datasets_dir, input_dataset_name + '_caesar')
    if os.path.isdir(output_path):
        return
    
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)

    input_path = os.path.join(datasets_dir, input_dataset_name)
    if not os.path.isdir(input_path):
        raise ValueError(f"Dataset {input_dataset_name} not found in {datasets_dir}")
    
    input_dataset = datasets.load_from_disk(input_path)
    cipher = _make_cipher(input_dataset['text'])
    output_dataset = input_dataset.map(lambda x: {'text': _apply_cipher(x['text'], cipher)})
    output_dataset.save_to_disk(output_path)


# Datasets

def save_imdb(n=2048, seed=1234, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    output_path = os.path.join(datasets_dir, 'imdb')
    if os.path.isdir(output_path):
        return
    
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    
    imdb = datasets.load_dataset('stanfordnlp/imdb', split='test', cache_dir=hf_cache_dir).shuffle(seed=seed).take(n)
    imdb.save_to_disk(output_path)


def save_random_strings(length=100, n=2048, seed=1234, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    output_path = os.path.join(datasets_dir, 'random_strings')
    if os.path.isdir(output_path):
        return
    
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"

    torch.manual_seed(seed)
    random_strings_list = []
    for _ in range(n):
        random_strings_list.append({'text': ''.join([chars[i] for i in torch.randint(len(chars), (length,))])})

    random_strings = datasets.Dataset.from_list(random_strings_list)
    random_strings.save_to_disk(output_path)


def save_web_text(max_length=10000, n=2048, seed=1234, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    output_path = os.path.join(datasets_dir, 'web_text')
    if os.path.isdir(output_path):
        return
    
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    
    owt = datasets.load_dataset('Skylion007/openwebtext', split='train', cache_dir=hf_cache_dir, trust_remote_code=True)
    owt = owt.shuffle(seed=seed).to_iterable_dataset().filter(lambda x: len(x['text']) < max_length).take(n)
    owt = datasets.Dataset.from_list(list(owt))
    owt.save_to_disk(output_path)


def save_book_translations(n=2048, seed=1234, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    output_path_en = os.path.join(datasets_dir, 'book_translations_en')
    output_path_de = os.path.join(datasets_dir, 'book_translations_de')
    if os.path.isdir(output_path_en) and os.path.isdir(output_path_de):
        return
    
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    
    books = datasets.load_dataset('Helsinki-NLP/opus_books', 'de-en', split='train', cache_dir=hf_cache_dir).shuffle(seed=seed).take(n)

    en_list = []
    de_list = []
    for tr in books['translation']:
        en_list.append({'text': tr['en']})
        de_list.append({'text': tr['de']})

    en_dataset = datasets.Dataset.from_list(en_list)
    en_dataset.save_to_disk(output_path_en)

    de_dataset = datasets.Dataset.from_list(de_list)
    de_dataset.save_to_disk(output_path_de)


def save_datasets(n=2048, seed=1234, datasets_dir=None, hf_cache_dir=None):
    datasets_dir, hf_cache_dir = _get_dirs(datasets_dir, hf_cache_dir)

    save_imdb(n=n, seed=seed, datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)
    save_random_strings(n=n, seed=seed, datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)
    save_web_text(n=n, seed=seed, datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)
    save_book_translations(n=n, seed=seed, datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)
    save_caesar('web_text', seed=seed, datasets_dir=datasets_dir, hf_cache_dir=hf_cache_dir)