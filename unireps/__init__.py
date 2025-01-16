from .paths import datasets_directory, hf_cache_directory, outputs_directory, set_datasets_directory, set_hf_cache_directory, set_outputs_directory
from .datasets import save_datasets, save_caesar, save_imdb, save_random_strings, save_web_text, save_book_translations, save_common_words
from .outputs import get_dataset_name, get_dataset_path, get_dataset, dataset_embs, generate_embeddings
from .representations import embed_dataset
from .similarity import embs_knn, mutual_knn, mutual_knn_baseline
from .visualizations import layer_by_layer_plot

__author__ = 'Christopher Wolfram'
