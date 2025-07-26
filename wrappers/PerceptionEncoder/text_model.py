from functools import partial
from logging import getLogger
from typing import Callable, Optional
import torch
from torch import nn
import torch.nn.functional as F
from core.vision_encoder.pe import Transformer


logger = getLogger()

class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 72,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 1280,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        output_tokens: bool = False,
        use_ln_post: bool = True,
        
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.pool_type = pool_type
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.layers = layers

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )

        if pool_type == "attn" or pool_type == "attn_eos":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:  # argmax
            self.attn_pool = None

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def load_ckpt(self, ckpt_path: str, verbose: bool = True):
        _sd = torch.load(ckpt_path, map_location='cpu')
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # Prendi solo i pesi relativi a TextTransformer
        text_sd = {}
        for k, v in _sd.items():
            if k.startswith("text_transformer."):
                text_sd[k.replace("text_transformer.", "", 1)] = v
            elif not ("visual." in k or k == "logit_scale"):  # fallback
                text_sd[k] = v

        m, u = self.load_state_dict(text_sd, strict=False)
        if verbose or (m or u):
            logger.info(f"Missing keys for TextTransformer: {m}")
            logger.info(f"Unexpected keys for TextTransformer: {u}")
            print(f"Missing keys for TextTransformer: {m}")
            print(f"Unexpected keys for TextTransformer: {u}")

    def build_cls_mask(self, text):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def text_global_pool(
        self, x, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"
    ):
        if pool_type == "first":
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == "last":
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == "argmax":
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, text):
        seq_len = text.shape[1]
        x = self.token_embedding(
            text
        ) 
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = self.text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled
