import torch
from torch import nn
from typing import Optional

from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel  
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from transformers.models.siglip.modeling_siglip import (
    SiglipPreTrainedModel,
    SiglipTextConfig,
    SiglipTextEmbeddings,
    SiglipEncoder,
    SiglipMultiheadAttentionPoolingHead
)

class CustomSiglipTextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.width = embed_dim
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"        

    def get_token_embedding_layer(self):
        return self.embeddings.token_embedding

    def prompt_forward(self, prompt, task_tokenized_prompts):        
        '''
            This method use the prompts embeddings directly instead of computing them using the embedding layer
        '''

        position_id = self.embeddings.position_ids
        pos = self.embeddings.position_embedding(position_id)

        hidden_states = prompt + pos    

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # The model uses the last token's hidden state, which may be padding.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return pooled_output


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # The model uses the last token's hidden state, which may be padding.        
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipTextModel(SiglipPreTrainedModel):
    config: SiglipTextConfig

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.width = config.hidden_size
        self.text_model = CustomSiglipTextTransformer(config)
        self.post_init()

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary into the text model."""
        return self.text_model.load_state_dict(state_dict, strict=strict)

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def prompt_forward(self, prompt, task_tokenized_prompts):
        return self.text_model.prompt_forward(prompt, task_tokenized_prompts)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
