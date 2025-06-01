# grpo/train_grpo.py

import os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel

from grpo_trainer import GRPOTrainer
from dataset_utils import GRPOJsonDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/grpo_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    policy_model = PeftModel.from_pretrained(base_model, config["lora_path"], is_trainable=True)

    dataset = GRPOJsonDataset(
        config["dataset_path"],
        tokenizer,
        max_length=config["max_seq_length"]
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["num_epochs"],
        logging_dir=config["logging_dir"],
        logging_steps=20,
        save_total_limit=2,
        save_steps=500,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=policy_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
