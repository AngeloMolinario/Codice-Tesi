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


class Siglip2Model(SiglipPreTrainedModel):

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
        text, normalize: bool = False
    ) -> torch.FloatTensor:

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=text,
            attention_mask=None,
            position_ids=None,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )

        pooled_output = text_outputs.pooler_output

        if normalize:
            pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)

        return pooled_output

    def get_image_features(
        self,
        image, normalize=False
    ) -> torch.FloatTensor:
        

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=image,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            interpolate_pos_encoding=None,
        )

        pooled_output = vision_outputs.pooler_output
        if normalize:
            pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)
            
        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> SiglipOutput:


        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            interpolate_pos_encoding=False,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
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
    