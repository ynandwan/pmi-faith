from transformers.generation_logits_process import (
    LogitsProcessorList,
    TopPLogitsWarper,
    LogitsProcessor,
)

import torch.nn.functional as F
import torch
from IPython.core.debugger import Pdb



class PMILogitsProcessor(LogitsProcessor):
    def __init__(
        self, top_p=1.0, add_log_prob=1, pmi_weight=0.5
    ):
        self.internal_top_p = TopPLogitsWarper(top_p=top_p)
        self.add_log_prob = add_log_prob
        self.pmi_weight = pmi_weight

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        log_prob = F.log_softmax(scores.float(), dim=-1)
        # top_p_mask has -inf at tokens that need to be masked. At other places,
        # its value is same as log_prob
        top_p_mask = self.internal_top_p(input_ids, log_prob)
        batch_size = log_prob.shape[0] // 2
        logprob_with_doc = log_prob[:batch_size]
        logprob_wo_doc = log_prob[batch_size:]
        # 1 whenever both lob probs are +inf or -inf.
        same_inf_mask = (
            (logprob_with_doc == -torch.inf) & (logprob_wo_doc == -torch.inf)
        ) | ((logprob_with_doc == torch.inf) & (logprob_wo_doc == torch.inf))
        # map 1 to - 1 and 0 to 1 by -2x + 1 so that this becomes a multiplicative mask
        same_inf_mask = -2.0 * same_inf_mask + 1.0
        pmi_adjusted_score = logprob_with_doc - same_inf_mask * logprob_wo_doc

        if (self.add_log_prob == 0) or (self.pmi_weight == 1.0):
            # convert non inf entries to 0 in top_p_mask so that they do not contribute in ranking
            top_p_mask[top_p_mask != -torch.inf] = 0.0

        if self.pmi_weight == 0:
            final_score = top_p_mask[:batch_size]
        elif self.pmi_weight == 1.0:
            final_score = pmi_adjusted_score + top_p_mask[:batch_size]
        else:
            final_score = (
                self.pmi_weight * pmi_adjusted_score
                + (1.0 - self.pmi_weight) * top_p_mask[:batch_size]
            )
        return final_score.repeat(2, 1)

