# dpo/dataset_utils.py

import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DPOJsonDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, max_length=512):
        with open(file_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        prompt_ids = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen, truncation=True, max_length=self.max_length, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected, truncation=True, max_length=self.max_length, add_special_tokens=False)

        return {
            "prompt_input_ids": torch.tensor(prompt_ids),
            "chosen_input_ids": torch.tensor(chosen_ids),
            "rejected_input_ids": torch.tensor(rejected_ids),
        }
