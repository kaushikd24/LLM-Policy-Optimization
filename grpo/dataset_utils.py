# grpo/dataset_utils.py

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class GRPOJsonDataset(Dataset):
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
        responses = sample["responses"]
        scores = sample["scores"]

        prompt_ids = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length, add_special_tokens=False)

        # For each response, tokenize prompt + response
        input_ids = []
        attn_masks = []
        for response in responses:
            resp_ids = self.tokenizer.encode(response, truncation=True, max_length=self.max_length, add_special_tokens=False)
            combined = prompt_ids + resp_ids
            pad_len = self.max_length - len(combined)
            combined = combined[:self.max_length] + [self.tokenizer.pad_token_id] * max(pad_len, 0)
            mask = [1] * min(len(combined), self.max_length) + [0] * max(pad_len, 0)
            input_ids.append(combined[:self.max_length])
            attn_masks.append(mask[:self.max_length])

        return {
            "input_ids": torch.tensor(input_ids),         # (K, T)
            "attention_mask": torch.tensor(attn_masks),   # (K, T)
            "scores": torch.tensor(scores, dtype=torch.float),  # (K,)
        }
