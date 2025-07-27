from logging import getLogger
import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from functools import partial
from core.vision_encoder.pe import TextTransformer, VisionTransformer
from core.vision_encoder.rope import Rope2D
from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG, fetch_pe_checkpoint
from torch.nn import functional as F
from dataclasses import asdict
import numpy as np

from wrappers.PerceptionEncoder.vision_model import VisionTransformer as CustomVisionTransformer
from wrappers.PerceptionEncoder.text_model import TextTransformer as CustomTextTransformer

logger = getLogger()

class PECore(nn.Module):
    def __init__(
        self,
        vision_cfg: PEConfig,
        text_cfg: PETextConfig,
        init_logit_scale: float = np.log(1 / 0.07),
        num_prompt: int = 0,  # Number of context prompt tokens to be prepended to the image patches
    ):
        super(PECore, self).__init__()
        self.visual = CustomVisionTransformer(**asdict(vision_cfg), num_prompt=num_prompt)
        self.image_size = self.visual.image_size  # For ease of use
        self.text_model = CustomTextTransformer(**asdict(text_cfg))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.dtype = self.text_model.token_embedding.weight.dtype  # Get the dtype from the text model

    def encode_image(self, image, normalize: bool = False):
        x = self.visual(image)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_text(self, text, normalize: bool = False):
        x = self.text_model(text)
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
    
    def load_logit_scale(self, ckpt_path: str):
        _sd = torch.load(ckpt_path, map_location='cpu')
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]

        # Cerca il parametro logit_scale
        if "logit_scale" in _sd:
            self.logit_scale.data.copy_(_sd["logit_scale"])
        elif "logit_scale" in _sd.get("module", {}):
            self.logit_scale.data.copy_(_sd["module"]["logit_scale"])

    def load_ckpt(self,
        name: str,
        checkpoint_path: Optional[str] = None,
    ):
        if name not in PE_VISION_CONFIG or name not in PE_TEXT_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")

        self.visual.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
        self.text_model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
        self.load_logit_scale(fetch_pe_checkpoint(name, checkpoint_path))

    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        num_prompt: int = 0,  # Number of context prompt tokens to be prepended to the image patches
    ):
        if name not in PE_VISION_CONFIG or name not in PE_TEXT_CONFIG:
            raise RuntimeError(f"{name} not found in configs.")

        model = cls(PE_VISION_CONFIG[name], PE_TEXT_CONFIG[name], num_prompt=num_prompt)
        if pretrained:
            model.visual.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
            model.text_model.load_ckpt(fetch_pe_checkpoint(name, checkpoint_path))
            model.load_logit_scale(fetch_pe_checkpoint(name, checkpoint_path))
        return model

