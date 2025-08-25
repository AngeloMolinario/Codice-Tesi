from torch import nn
from typing import Optional
import torch

from wrappers.promptopt.prompt_learner import VisionPromptLearner

from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel  # base for SiglipPreTrainedModel

from transformers.models.siglip.modeling_siglip import (
    SiglipPreTrainedModel,
    SiglipVisionConfig,
    SiglipVisionEmbeddings,
    SiglipEncoder,
    SiglipMultiheadAttentionPoolingHead,
)



class CustomSiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig, num_prompt=0):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)

        # Add the Prompt Learner if needed
        self.num_prompt = num_prompt
        if num_prompt > 0:
            self.prompt_learner = VisionPromptLearner(num_prompt=num_prompt, emb_size=config.hidden_size, is_cls_present=False)  # Uncomment if using a prompt learner

        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        
        

        if self.num_prompt > 0:
            hidden_states = self.prompt_learner(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        if self.use_head:         
            pooler_output = self.head(last_hidden_state)
        else:
            pooler_output = None

        return pooler_output

class SiglipVisionModel(SiglipPreTrainedModel):
    config: SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig, num_prompt=0):
        super().__init__(config)

        self.vision_model = CustomSiglipVisionTransformer(config, num_prompt=num_prompt)


        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_prompt=0, *model_args, **kwargs):
        """Load pre-trained weights for the vision model."""
        config = SiglipVisionConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(config, num_prompt=num_prompt)
        model.load_state_dict(torch.load(pretrained_model_name_or_path))
        return model

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary into the vision model."""
        return self.vision_model.load_state_dict(state_dict, strict=strict)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
