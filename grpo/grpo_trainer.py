# grpo/grpo_trainer.py

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Tuple, Optional


def compute_grpo_kl_loss(
    seq_logprobs: torch.Tensor,
    scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL(π_policy || π_target), where π_target is softmax over human scores.
    π_target is detached.
    """
    log_p = torch.log_softmax(seq_logprobs, dim=-1)
    log_q = torch.log_softmax(scores, dim=-1).detach()
    return F.kl_div(log_p, log_q, log_target=True, reduction="batchmean")


class GRPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        input_ids = inputs["input_ids"]           # (B, K, T)
        attention_mask = inputs["attention_mask"] # (B, K, T)
        scores = inputs["scores"]                 # (B, K)

        B, K, T = input_ids.shape
        input_ids = input_ids.view(B * K, T)
        attention_mask = attention_mask.view(B * K, T)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits                   # (B*K, T, vocab)
        log_probs = -F.cross_entropy(
            logits.transpose(1, 2), input_ids, reduction='none'
        )
        seq_logprobs = log_probs.sum(dim=1).view(B, K)

        loss = compute_grpo_kl_loss(seq_logprobs, scores)
        return (loss, {"loss": loss}) if return_outputs else loss
