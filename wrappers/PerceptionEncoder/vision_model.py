import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from functools import partial
from core.vision_encoder.pe import AttentionPooling, Transformer
from core.vision_encoder.rope import Rope2D
from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG, fetch_pe_checkpoint
from logging import getLogger
from torch.nn import functional as F
from dataclasses import asdict

from wrappers.PerceptionEncoder.CustomRope2D import CustomRope2D
from wrappers.promptopt.prompt_learner import VisionPromptLearner

logger = getLogger()


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        image_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
        num_prompt: int = 0, # Number of context prompt tokens to be prepended to the image patches
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size

        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size

        self.num_prompt = num_prompt  # Number of prompt tokens, can be adjusted as needed

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.rope = (
                CustomRope2D(
                    dim=width // heads,
                    num_prompt=num_prompt,
                    use_cls_token=self.use_cls_token,
                )
                if self.use_rope2d
                else None
            )
        

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None


        self.init_tensors()


        # Prompt tuning learnable parameters
        if self.num_prompt > 0:
            logger.info(f"Using {self.num_prompt} prompt tokens.")
            self.prompt_learner = VisionPromptLearner(self.num_prompt, self.width, self.use_cls_token)
        else:
            logger.info("No prompt tokens used.")
            self.prompt_learner = None

    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        self.rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2, self.width
                )
            )

        if self.proj_dim is not None:
            self.proj = nn.Parameter(
                init_scale * torch.randn(self.width, self.proj_dim)
            )


    def load_ckpt(self, ckpt_path: str, verbose: bool = True):
        _sd = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        if any(k.startswith("visual.") for k in _sd):
            _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}

        m, u = self.load_state_dict(_sd, strict=False)

        if verbose or (m or u):
            logger.info(f"Missing keys for loading vision encoder: {m}")
            logger.info(f"Unexpected keys for loading vision encoder: {u}")
            print(f"Missing keys for loading vision encoder: {m}")
            print(f"Unexpected keys for loading vision encoder: {u}")
        return

    def truncate(self, layer_idx: int):
        """ Delete layers so the last layer is the given layer index. """
        self.transformer.truncate(layer_idx)
        self.layers = self.transformer.layers

    def get_config(self):
        config = {
            "patch_size": self.patch_size,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "mlp_ratio": self.mlp_ratio,
            "act_layer": self.act_layer,
            "norm_layer": self.norm_layer,
            "use_ln_pre": self.use_ln_pre,
            "use_ln_post": self.use_ln_post,
            "ls_init_value": self.ls_init_value,
            "drop_path": self.drop_path,
            "image_size": self.image_size,
            "use_abs_posemb": self.use_abs_posemb,
            "use_rope2d": self.use_rope2d,
            "use_cls_token": self.use_cls_token,
            "output_dim": self.output_dim,
            "attn_pooler_heads": self.attn_pooler_heads,
            "pool_type": self.pool_type,
            "num_prompt": self.num_prompt,  # Number of prompt tokens
        }
        return config
    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        **kwdargs
    ):
        if name not in PE_VISION_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")
    
        args = asdict(PE_VISION_CONFIG[name])
        args.update(kwdargs)
        
        model = cls(**args)
        if pretrained:
            model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
        
        return model
    
    @classmethod
    def available_configs(cls):
        return list(PE_VISION_CONFIG.keys())


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable=enable)

    def _sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def _pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False
    ):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)
        
        # If self.num_prompt > 0 than use the PromptLearner to prepend prompt tokens
        if self.num_prompt > 0:
            x = self.prompt_learner(x)

        if self.use_rope2d:
            self.rope.update_grid(x.device, grid_h, grid_w)

        
        x = self.ln_pre(x)
        
        x = self.transformer(x, layer_idx=layer_idx)

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.forward_features(x, norm=True, **kwargs)
        x = self._pool(x)

        if self.proj_dim is not None:
            x = x @ self.proj

        return x

