from torch import nn
from helpers.utils import load_yaml
import torch
from typing import Optional, List

from Models.LLaVA.llava.model.llava_arch import LlavaMetaModel
from Models.LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

class BaselinePulseModel(nn.Module):
    def __init__(self, model_args, processor):
        super().__init__()
        self.config_ = load_yaml(model_args.model_config_path)
        self.model_pretrained_path = self.config_.get("model_pretrained")

        self.model_pretrained = LlavaLlamaForCausalLM.from_pretrained(
            self.model_pretrained_path,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )
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

        if pixel_values is not None and pixel_values.device.type == "cuda":
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
            pixel_values=None,
            attention_mask=None,
            max_new_tokens=50,
            **kwargs
    ):
        kwargs.pop('num_items_in_batch', None)

        if input_ids is None:
            raise ValueError("❌ input_ids is None — le processor n’a pas généré d’input_ids.")

        if pixel_values is not None and pixel_values.device.type == "cuda":
            pixel_values = pixel_values.to(torch.bfloat16)

        # ✅ PATCH pour LLaVA
        return self.model_pretrained.generate(
            inputs=input_ids,  # ✅ LLaVA attend `inputs=...`
            images=pixel_values,  # ✅ LLaVA attend `images=...`
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )