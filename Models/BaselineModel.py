from torch import nn
from helpers.utils import load_yaml
from transformers import AutoModelForImageTextToText
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

class BaselineVLModel(nn.Module):
    def __init__(self, model_args, processor):
        super().__init__()
        self.config_ = load_yaml(model_args.model_config_path)
        self.model_pretrained_path = self.config_.get("model_pretrained")
        attn_impl = self.config_.get("attn_implementation", "eager")

        self.model_pretrained = AutoModelForImageTextToText.from_pretrained(
            self.model_pretrained_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        #print(f"[DEBUG] Model loaded with FlashAttention: {attn_impl}, using bfloat16.")
        self.processor = processor

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ):
        kwargs.pop('num_items_in_batch', None)

        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        return self.model_pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            **kwargs
        )

    def generate(
        self,
        input_ids=None,
        max_new_tokens=50,
        **kwargs
    ):
        kwargs.pop('num_items_in_batch', None)
        output = self.model_pretrained.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        return output[:, input_ids.shape[1]:]
