# dpo/eval.py or grpo/eval.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model

def run_inference(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    MODEL_PATH = "model/base_gpt2"
    LORA_PATH = "outputs/dpo_lora_output"  # or outputs/grpo_lora_output if you've trained GRPO

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(MODEL_PATH, LORA_PATH)

    prompt = "Human: What is the meaning of life?\n\nAssistant:"
    response = run_inference(model, tokenizer, prompt)
    print("\n=== Model Response ===\n")
    print(response)
