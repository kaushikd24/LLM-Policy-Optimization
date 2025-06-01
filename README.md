# 🧠 DPO & GRPO: Advanced Policy Optimization for Language Models

Welcome to my DPO & GRPO project, a comprehensive framework for optimizing language models using Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), from Deepseek. This project leverages Hugging Face Transformers and PEFT (LoRA) to provide efficient and scalable solutions for preference learning and policy optimization.

---

### Overview

This repository contains two main components:

1. **Direct Preference Optimization (DPO)**: A method for training language models directly from preference pairs using a classification-style loss, improving sample quality and alignment in tasks like summarization, dialogue, and sentiment control.

2. **Group Relative Policy Optimization (GRPO)**: An extension of PPO that handles more complex preference structures, particularly suited for tasks like instruction tuning and mathematical reasoning, and also reduces training resources.

---

### Project Structure

```
dpo_grpo_rlhf/
├── data/
│   ├── raw/
│   ├── processed/
│   └── prompts.json
│
├── model/
│   └── base_gpt2/
│
├── configs/
│   ├── dpo_config.yaml
│   └── grpo_config.yaml
│
├── dpo/
│   ├── train_dpo.py
│   ├── dpo_trainer.py
│   ├── dataset_utils.py
│   ├── eval.py
│   └── README.md
│
├── grpo/
│   ├── train_grpo.py
│   ├── grpo_trainer.py
│   ├── dataset_utils.py
│   ├── eval.py
│   └── README.md
│
├── scripts/
│   └── download_model.py
│
├── requirements.txt
└── README.md
```

---

### Getting Started

To get started with either DPO or GRPO, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

### Running DPO

Configure your settings in `configs/dpo_config.yaml` and run:

```bash
python dpo/train_dpo.py
```

### Running GRPO

Configure your settings in `configs/grpo_config.yaml` and run:

```bash
python grpo/train_grpo.py
```

---

### Learn More

- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

---

### License

This project is licensed under the MIT License.

---

### Acknowledgements

- DPO: Based on the original Stanford DPO paper by Rafailov et al. (2023)
- GRPO: Based on the original DeepSeekMath Reasoning paper by the DeepSeek team.
- HuggingFace team and PEFT for making adapters dead simple

