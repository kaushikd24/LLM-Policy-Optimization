# DPO & GRPO: Advanced Policy Optimization for Language Models

Welcome to the DPO & GRPO project, a comprehensive framework for optimizing language models using Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). This project leverages Hugging Face Transformers and PEFT (LoRA) to provide efficient and scalable solutions for preference learning and policy optimization.

---

### Overview

This repository contains two main components:

1. **Direct Preference Optimization (DPO)**: A method for training language models directly from preference pairs using a classification-style loss, improving sample quality and alignment in tasks like summarization, dialogue, and sentiment control.

2. **Group Relative Policy Optimization (GRPO)**: An extension of DPO that handles more complex preference structures, particularly suited for tasks like instruction tuning and mathematical reasoning.

- Kindly open the DPO and GRPO folders to learn more about how the algorithms work !

---

## Project Structure

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
│   └── convert_raw_to_processed.py
│
├── requirements.txt
└── README.md
```

---

## Getting Started

To run this project on your device, clone the repository using the following command:

```bash
git clone https://github.com/kaushikd24/LLM-Policy-Optimization.git
cd LLM-Policy-Optimization
```

Ensure you have the necessary dependencies installed:

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

## Learn More

- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Based on the original Stanford DPO paper by Rafailov et al. (2023)
- HuggingFace 🤗 team and PEFT for making adapters dead simple
- Thanks to the contributors and the open-source community for their support and resources.

