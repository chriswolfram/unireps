import sys
import huggingface_hub

if __name__ == "__main__":
     # Load command-line arguments
    model_list = sys.argv[1]
    cache_dir = sys.argv[2]
    
    # Authenticate for HF
    huggingface_hub.login(new_session=False)

    # Cache models
    with open(model_list) as f:
        for model_name_nl in f:
            model_name = model_name_nl.rstrip()
            print("Downloading", model_name)
            huggingface_hub.snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                force_download=False,
                allow_patterns="*.safetensors"
            )