import torch

def embed_dataset(dataset, model, tokenizer, use_chat_template=False, text_key='text'):
    """
    Embed a dataset using a specified model and tokenizer.
    Args:
        dataset (datasets.Dataset): The dataset to embed.
        model (transformers.PreTrainedModel): The pre-trained model to use for embedding.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for processing the text.
        use_chat_template (bool, optional): Whether to use a chat template for tokenization. Defaults to False.
        text_key (str, optional): The key in the dataset examples that contains the text to embed. Defaults to 'text'.
    Returns:
        datasets.Dataset: The dataset with computed embeddings added.
    """

    # compute_embeddings is mapped over the input dataset
    def compute_embeddings(example):

        ### Tokenize
        if use_chat_template:
            messages = [{'role': 'user', 'content': example[text_key]}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, truncation=True, return_tensors='pt')
            input_ids = input_ids.to(model.device)
        else:
            tokens = tokenizer(example[text_key], truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(model.device)
        
        ### Run the model
        with torch.no_grad():
            model_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        
        ### Extract hidden states
        # The hidden states are cast to float32 becauase it avoids some numerical issues, and everything
        # will have to be cast eventually because Arrow doesn't currently support types like bfloat16.
        layer_token_embeddings = torch.stack(model_out.hidden_states)[:,0].float()

        ### Compute embeddings from hidden states
        # (Outputs are moved to the CPU to avoid memory problems on the GPU.)
        layer_last_embeddings = layer_token_embeddings[:,-1].cpu()
        layer_mean_embeddings = layer_token_embeddings.mean(1).cpu()

        return {
            'layer_last_embeddings': layer_last_embeddings,
            'layer_mean_embeddings': layer_mean_embeddings
        }

    # TODO: This currently sets new_fingerprint because otherwise `map` appears to hash compute_embeddings which includes the entire model!
    # This does not use batching. In experiments, batching (somehow) slightly slowed it down across models and hardware!
    embeddings = dataset.map(compute_embeddings, new_fingerprint='fingerprint', load_from_cache_file=False)
    embeddings.set_format('torch')
    
    return embeddings