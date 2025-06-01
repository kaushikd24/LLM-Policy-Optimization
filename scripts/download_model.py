from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import os

# Show HuggingFace file progress bar
logging.set_verbosity_info()

def download_and_save(model_name: str, save_dir: str):
    print(f"\nDownloading and caching model: {model_name}")
    
    try:
        # Download and cache
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Save locally
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        print(f"\nModel saved to: {save_dir}")
        return True
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Please try again.")
        return False
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return False

if __name__ == "__main__":
    model_name = "openai-community/gpt2"   # or mistralai/Mistral-7B-v0.1
    save_dir = "models/base_gpt2"
    download_and_save(model_name, save_dir) 