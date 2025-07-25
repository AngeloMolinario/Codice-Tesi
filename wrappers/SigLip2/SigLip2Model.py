from typing import Optional
import torch
from torch import nn
from .vision import SiglipVisionModel
from .text import SiglipTextModel

from transformers.models.siglip.modeling_siglip import (
    SiglipPreTrainedModel,
)
from transformers.models.siglip.configuration_siglip import (
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import SiglipOutput


class SiglipModel(SiglipPreTrainedModel):
    config: SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)

        if not isinstance(config.text_config, SiglipTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = SiglipTextModel._from_config(text_config)
        vision_model = SiglipVisionModel._from_config(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs.pooler_output

        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs.pooler_output

        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> SiglipOutput:

        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
    



if __name__ == "__main__":
    # Example usage
    model = SigLipModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    print("#"*10 + " Model Configuration " + "#"*10)
    print(model.config)
    print("#"*10 + " Model Summary " + "#"*10)
    