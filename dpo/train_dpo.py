# dpo/train_dpo.py

import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

from dpo_trainer import DPOTrainer
from dataset_utils import DPOJsonDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/dpo_config.yaml")
    model_path = config["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # Load base model and apply LoRA
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["c_attn"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy_model = get_peft_model(base_model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(model_path)

    dataset = DPOJsonDataset(
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

    trainer = DPOTrainer(
        model=policy_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        ref_model=ref_model,
        beta=config["beta"]
    )

    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
