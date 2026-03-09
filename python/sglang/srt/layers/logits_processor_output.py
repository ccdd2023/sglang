import dataclasses
from typing import Any, Dict, List, Optional, Union

import torch


@dataclasses.dataclass
class LogitsProcessorOutput:
    # The logits of the next tokens. shape: [#seq, vocab_size]
    next_token_logits: Optional[torch.Tensor]
    # Used by speculative decoding (EAGLE)
    hidden_states: Optional[torch.Tensor] = None

    # The logprobs and ids of output tokens.
    next_token_logprobs: Optional[torch.Tensor] = None
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None

    # Prefill-only logprob outputs.
    input_token_logprobs: Optional[torch.Tensor] = None
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None

    # Diffusion LLM only.
    full_logits: Optional[torch.Tensor] = None

    # Customized outputs.
    customized_info: Optional[Dict[str, List[Any]]] = None
    mm_input_embeds: Optional[torch.Tensor] = None
