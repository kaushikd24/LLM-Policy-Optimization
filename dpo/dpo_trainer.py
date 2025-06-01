# dpo/dpo_trainer.py

import torch
from torch.nn import functional as F
from transformers import Trainer


class DPOTrainer(Trainer):
    def __init__(self, ref_model=None, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta

        if self.ref_model:
            self.ref_model.eval()  # Make sure ref model doesn't train

    def compute_loss(self, model, inputs, return_outputs=False):
        prompt_input_ids = inputs["prompt_input_ids"]
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]

        # Concatenate prompt + response
        chosen_inputs = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
        rejected_inputs = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)

        # Get log probs from policy model
        policy_chosen_logps = self._get_logps(model, chosen_inputs)
        policy_rejected_logps = self._get_logps(model, rejected_inputs)

        # Get log probs from reference model (no LoRA)
        with torch.no_grad():
            ref_chosen_logps = self._get_logps(self.ref_model, chosen_inputs)
            ref_rejected_logps = self._get_logps(self.ref_model, rejected_inputs)

        # DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        loss = losses.mean()

        return (loss, {"loss": loss}) if return_outputs else loss

    def _get_logps(self, model, input_ids):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        log_probs = -F.cross_entropy(
            output.logits.transpose(1, 2), input_ids, reduction='none'
        )
        # Only take logprobs for response part
        return log_probs.sum(dim=1)
