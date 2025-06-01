# GRPO: Group Relative Policy Optimization

### Overview

**Group Relative Policy Optimization (GRPO)** is a lightweight and efficient reinforcement learning (RL) algorithm designed to fine-tune large language models (LLMs) *without* the computational overhead of a critic. It's particularly suited for tasks like instruction tuning and mathematical reasoning, where reward signal comparison across groups of outputs is more informative than scalar-valued feedback.

This repo implements the GRPO algorithm with HuggingFace Transformers + LoRA (via PEFT), adapted for autoregressive language modeling tasks like math instruction tuning, preference learning, and beyond.

> GRPO was originally introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), where it helped outperform all open-source 7B–70B models on the MATH benchmark — with fewer parameters and less compute than PPO.

---

### Key Features

* **KL-based loss over relative sequence scores** — no critic network needed.
* **LoRA-compatible** via HuggingFace PEFT.
* Drop-in support for any `AutoModelForCausalLM`-style model (e.g., GPT-2, LLaMA).
* Built-in dataset support for GRPO-style JSON formats.
* Optimized for training efficiency using `Trainer`.

---

### Project Structure

```bash
grpo/
├── grpo_trainer.py        # Custom HuggingFace Trainer with GRPO loss
├── train_grpo.py          # Training script (uses YAML config)
├── dataset_utils.py       # Dataset class to load and batch GRPO JSON data
├── configs/
│   └── grpo_config.yaml   # Example training config
├── models/                # (Optional) Pretrained or LoRA model dirs
└── data/                  # Your dataset (JSON format with scores)
```

---

### How GRPO Works (TL;DR)

Instead of optimizing a scalar reward like PPO does with a critic, **GRPO**:

1. Samples multiple completions per prompt.
2. Computes log-probabilities over these sequences.
3. Uses human or model-provided **group scores** to compute a soft target distribution.
4. Applies a **KL divergence** loss between the model's sequence probabilities and the score-based target.

This avoids the need for a value network and is more stable under sparse rewards.

---

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

You'll need:

* `transformers`
* `peft`
* `datasets`
* `torch`
* `PyYAML`
* (Optional) `accelerate` if you plan to scale

---

### Example: Running GRPO Training

Configure your settings in `configs/grpo_config.yaml`. Then run:

```bash
python grpo/train_grpo.py
```

Your config YAML should look like this:

```yaml
model_path: gpt2
lora_path: path/to/lora/checkpoint
dataset_path: data/grpo_dataset.json
output_dir: grpo_outputs/
max_seq_length: 512
batch_size: 8
num_epochs: 3
logging_dir: logs/
```

---

### Example Dataset Format

The dataset should be a JSONL file where each entry contains multiple completions per prompt with associated scores:

```json
{
  "prompt": "Prove that the square root of 2 is irrational.",
  "completions": [
    {"text": "Assume sqrt(2) is rational...", "score": 3.2},
    {"text": "Using Pythagoras theorem...", "score": 1.8},
    ...
  ]
}
```

Each group is tokenized into a batch of shape `(B, K, T)` — Batch size × Completions × Tokens — for GRPO loss.

---

### The GRPO Loss

```python
KL(π_policy || π_target), where
π_policy = softmax(model log probs)
π_target = softmax(human/model scores)
```

We apply this KL loss over sequence-level log-probs:

```python
loss = KLDiv(log_p, log_q)
```

Where `log_q` is detached to avoid backprop through the reward function.

---

### Logging + Saving

* Logs every 20 steps (`logging_steps: 20`)
* Saves checkpoints every 500 steps
* Keeps the latest 2 checkpoints

Modify `TrainingArguments` in `train_grpo.py` if needed.

---

### Notes

* Make sure your tokenizer has a valid `pad_token_id` (we default to EOS).
* LoRA weights must be made trainable (`is_trainable=True`).
* Mixed-precision (fp16/bf16) training is easy to add via `TrainingArguments`.

---

### Paper

> *Group Relative Policy Optimization (GRPO)* was proposed in
> [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
> (Zhihong Shao et al., 2024)

---

### Related Work

* [DPO (Direct Preference Optimization)](https://arxiv.org/abs/2305.18290)
* [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347)
* [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
* [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)

---

### Files
- `train_grpo.py`: Main training script for GRPO.
- `grpo_trainer.py`: Custom trainer class for GRPO.
- `dataset_utils.py`: Utilities for preprocessing and loading datasets.
- `eval.py`: Evaluation utilities for assessing the performance of the GRPO model.

### Getting Started
To get started with training a model using GRPO, you can run the `train_grpo.py` script. Make sure to configure your environment and datasets appropriately.

### Requirements
Ensure all dependencies are installed as listed in the root `requirements.txt` file.

#### Acknowledgments
- Thanks to the contributors and the open-source community for their support and resources.

# GRPO Training (LoRA + GPT-2)

This directory contains the GRPO training pipeline using HuggingFace + PEFT + LoRA.

### Run

```bash
python grpo/train_grpo.py --config configs/grpo_config.yaml
