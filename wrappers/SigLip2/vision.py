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


class SiglipEncoderWithPrompts(SiglipEncoder):
    """SigLIP encoder that injects learnable prompts at each transformer layer."""

    def __init__(self, config: SiglipVisionConfig, num_prompts: int):
        super().__init__(config)
        self.num_prompts = num_prompts
        self.deep_prompt_embeddings = nn.Parameter(
            torch.randn(config.num_hidden_layers, num_prompts, config.hidden_size)
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer_idx, encoder_layer in enumerate(self.layers):
            if self.num_prompts > 0:
                prompts = self.deep_prompt_embeddings[layer_idx]
                prompts = prompts.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
                hidden_states = torch.cat([prompts, hidden_states], dim=1)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if self.num_prompts > 0:
                hidden_states = hidden_states[:, self.num_prompts :, :]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)



class CustomSiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig, num_prompt=0, deep_prompt: bool = False):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)

        self.num_prompt = num_prompt
        self.deep_prompt = deep_prompt

        if num_prompt > 0 and not deep_prompt:
            self.prompt_learner = VisionPromptLearner(
                num_prompt=num_prompt, emb_size=config.hidden_size, is_cls_present=False
            )

        if num_prompt > 0 and deep_prompt:
            self.encoder = SiglipEncoderWithPrompts(config, num_prompt)
        else:
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

        if self.num_prompt > 0 and not self.deep_prompt:
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

    def __init__(self, config: SiglipVisionConfig, num_prompt=0, deep_prompt: bool = False):
        super().__init__(config)

        self.vision_model = CustomSiglipVisionTransformer(
            config, num_prompt=num_prompt, deep_prompt=deep_prompt
        )


        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, num_prompt=0, deep_prompt: bool = False, *model_args, **kwargs
    ):
        """Load pre-trained weights for the vision model."""
        config = SiglipVisionConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(config, num_prompt=num_prompt, deep_prompt=deep_prompt)
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
