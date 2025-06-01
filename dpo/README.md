# 🧠 Direct Preference Optimization (DPO) Training Pipeline

> "Your Language Model is Secretly a Reward Model." — [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290)

This repo implements **Direct Preference Optimization (DPO)** using Hugging Face Transformers + PEFT (LoRA), allowing stable and scalable preference learning — without reinforcement learning or reward models.

DPO trains a policy model directly from preference pairs `(prompt, chosen_response, rejected_response)` using a **classification-style loss**, improving sample quality and alignment in tasks like summarization, dialogue, and sentiment control.

---

## 🗂 Directory Structure

```
dpo/
├── dpo_trainer.py       # Custom DPOTrainer that computes DPO loss
├── train_dpo.py         # End-to-end training script using Hugging Face Trainer API
├── dataset_utils.py     # Dataset class for loading JSON preference pairs
├── configs/             
│   └── dpo_config.yaml  # YAML config for training hyperparameters
```

---

## 🧮 Key Components

### 🔧 `DPOTrainer` (`dpo_trainer.py`)

Custom trainer class that implements the DPO loss:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \cdot \left[ \log\frac{\pi(a^+|s)}{\pi_{\text{ref}}(a^+|s)} - \log\frac{\pi(a^-|s)}{\pi_{\text{ref}}(a^-|s)} \right] \right)
$$

Where:

* `a⁺` is the **chosen** response
* `a⁻` is the **rejected** response
* `π` is the current policy (LoRA fine-tuned)
* `π_ref` is the reference model (frozen base model)
* `β` is a hyperparameter controlling KL-penalty

### 🛠️ `train_dpo.py`

* Loads config, tokenizer, and base model
* Applies LoRA to the policy model
* Loads preference dataset
* Initializes `DPOTrainer` and begins training

---

## 🧪 Run Training

> Make sure `configs/dpo_config.yaml` is correctly filled.

```bash
python dpo/train_dpo.py
```

---

## 🧾 Sample Config (`configs/dpo_config.yaml`)

```yaml
model_path: "gpt2"
dataset_path: "data/preferences.json"
output_dir: "checkpoints/dpo"
batch_size: 4
num_epochs: 3
logging_dir: "logs"
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
max_seq_length: 512
beta: 0.1
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You'll need:

* `transformers`
* `peft`
* `datasets`
* `torch`
* `pyyaml`

---

## 🧠 Theory: Why DPO?

DPO sidesteps RL by **reparameterizing the reward model** using log-probability ratios between policy and reference model. This makes DPO:

* Simpler than PPO (no sampling loops, no reward model)
* More stable (no value function)
* Empirically strong — often **outperforms PPO** on summarization and dialogue tasks

📚 Paper: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

---

## 🪪 License

MIT License

---

## 🙏 Acknowledgements

* Based on the original Stanford DPO paper by Rafailov et al. (2023)
* HuggingFace 🤗 team and PEFT for making adapters dead simple

## Files
- `train_dpo.py`: Main training script for DPO.
- `dpo_trainer.py`: Custom trainer class for DPO.
- `dataset_utils.py`: Utilities for preprocessing and loading datasets.
- `eval.py`: Evaluation utilities for assessing the performance of the DPO model.

## Getting Started
To get started with training a model using DPO, you can run the `train_dpo.py` script. Make sure to configure your environment and datasets appropriately.

## Requirements
Ensure all dependencies are installed as listed in the root `requirements.txt` file.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to the contributors and the open-source community for their support and resources.

# DPO Training (LoRA + GPT-2)

This directory contains the DPO training pipeline using HuggingFace + PEFT + LoRA.

## Run

```bash
python dpo/train_dpo.py --config configs/dpo_config.yaml
