import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from functools import partial
from core.vision_encoder.pe import TextTransformer
from core.vision_encoder.rope import Rope2D
from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG, fetch_pe_checkpoint
from logging import getLogger
from torch.nn import functional as F
from dataclasses import asdict
import numpy as np
from wrappers.PerceptionEncoder.vision_model import VisionTransformer

logger = getLogger()


class PECore(TextTransformer):
    def __init__(
        self,
        vision_cfg: PEConfig,
        text_cfg: PETextConfig,
        init_logit_scale: float = np.log(1 / 0.07)
    ):
        super(PECore, self).__init__(**asdict(text_cfg))
        self.visual = VisionTransformer(**asdict(vision_cfg))
        self.image_size = self.visual.image_size  # For ease of use
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)


    def encode_image(self, image, normalize: bool = False):
        x = self.visual(image)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_video(self, video, normalize: bool = False): # b n c h w
        b, n, c, h, w = video.shape
        frms = video.reshape(b * n, c, h, w)
        frm_feats = self.encode_image(frms, normalize=normalize)
        video_feats = frm_feats.reshape(b, n, -1)
        video_feats = video_feats.mean(dim=1)
        return video_feats

    def encode_text(self, text, normalize: bool = False):
        x = super().forward(text)
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        return image_features, text_features, self.logit_scale.exp()
    

    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None  # To load your own
    ):
        if name not in PE_VISION_CONFIG or name not in PE_TEXT_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")
    
        model = cls(PE_VISION_CONFIG[name], PE_TEXT_CONFIG[name])
        if pretrained:
            model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
        
        return model

    @classmethod
    def available_configs(cls):
        return [k for k in PE_VISION_CONFIG if k in PE_TEXT_CONFIG]