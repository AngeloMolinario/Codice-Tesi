from logging import getLogger
import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from functools import partial
from core.vision_encoder import transforms
from core.vision_encoder.pe import TextTransformer, VisionTransformer
from core.vision_encoder.rope import Rope2D
from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG, fetch_pe_checkpoint
from torch.nn import functional as F
from dataclasses import asdict
import numpy as np
import os


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

    def get_image_features(self, image, normalize: bool = False):
        x = self.visual(image)
        return F.normalize(x, dim=-1) if normalize else x

    def get_text_features(self, text, normalize: bool = False):
        x = self.text_model(text)
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.get_image_features(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.get_text_features(text, normalize=True) if text is not None else None
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

    def save(self, save_path: str, text_features: torch.Tensor = None, text_features_path: str = None):
        """
        Salva solo i parametri necessari per l'inferenza (vision encoder + logit_scale)
        e opzionalmente le text features già calcolate in un file separato.

        Args:
            save_path: percorso dove salvare i parametri del vision encoder e logit_scale
            text_features: (opzionale) tensore di text features già calcolate da salvare
            text_features_path: (opzionale) dove salvare le text features, richiesto se text_features è fornito
        """
        import torch
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = {}

        # Salva tutti i parametri del vision encoder (inclusi eventuali VPT)
        for name, param in self.visual.state_dict().items():
            state_dict[f'visual.{name}'] = param

        # Salva il logit_scale
        state_dict['logit_scale'] = self.logit_scale.detach().cpu()

        torch.save(state_dict, save_path)
        print(f"Salvati parametri vision encoder e logit_scale in {save_path}")

        # Salva opzionalmente le text features
        if text_features is not None and text_features_path is not None:
            os.makedirs(os.path.dirname(text_features_path), exist_ok=True)
            torch.save({'text_features': text_features.cpu()}, text_features_path)
            print(f"Text features salvate in {text_features_path} (shape: {text_features.shape})")

    @classmethod
    def load_saved_model(cls, save_path: str, name: str = "PE-Core-B16-224"):
        """
        Carica un modello PECore salvato.
        Prima carica i pesi pretrained, poi sovraccarica con quelli salvati.
        
        Args:
            save_path: percorso del modello salvato
            name: nome della configurazione per i pesi pretrained
        """
        checkpoint = torch.load(save_path, map_location='cpu')
        
        if checkpoint['model_type'] == 'PECore_full':
            # Modello completo salvato
            config = checkpoint['config']
            model = cls.from_config(name, pretrained=False, num_prompt=config.get('num_prompt', 0))
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Solo nuovi pesi salvati - carica prima pretrained poi sovraccarica
            config = checkpoint['config']
            # 1. Carica modello pretrained
            model = cls.from_config(name, pretrained=True, num_prompt=config.get('num_prompt', 0))
            # 2. Sovraccarica con i nuovi pesi salvati
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model


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


from ..promptopt.prompt_learner import VisionPromptLearner

class PECore_Vision(nn.Module):
    def __init__(self, vision_cfg, num_prompt: int = 0, num_classes: tuple = (9, 2, 7), num_vpt: int = 1):
        super().__init__()
        self.visual = CustomVisionTransformer(**asdict(vision_cfg), num_prompt=num_prompt)
        self.image_size = self.visual.image_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.num_classes = num_classes
        self.num_prompt = num_prompt
        # Lista di prompt_learner esterni
        self.vpt_list = nn.ModuleList([
            VisionPromptLearner(num_prompt, 768, self.visual.use_cls_token)
            for _ in range(num_vpt)
        ])
        self.register_buffer('text_features', torch.empty(0))

    def load_vpt_checkpoints(self, vpt_checkpoint_paths):
        """
        Carica ogni checkpoint VPT in uno dei VisualPromptTuning istanziati nella lista.
        """
        assert len(vpt_checkpoint_paths) == len(self.vpt_list), "Numero VPT checkpoint deve corrispondere a num_vpt"
        for i, path in enumerate(vpt_checkpoint_paths):
            checkpoint = torch.load(path, map_location='cpu')
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            # Cerca chiavi relative al prompt learner
            vpt_state_dict = {}
            for name, param in state_dict.items():
                if "prompt_learner" in name and not name.startswith("visual."):
                    vpt_state_dict[name.replace("prompt_learner.", "")] = param
                elif "visual.prompt_learner" in name:
                    vpt_state_dict[name.replace("visual.prompt_learner.", "")] = param
            self.vpt_list[i].load_state_dict(vpt_state_dict, strict=True)
            print(f"VPT {i} caricato da {path}")

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        visual_state = {k.replace('visual.', ''): v for k, v in state.items() if k.startswith('visual.')}
        self.visual.load_state_dict(visual_state, strict=False)
        if 'logit_scale' in state:
            self.logit_scale.data.copy_(state['logit_scale'])
        print(f"Pesi VisionEncoder e logit_scale caricati da {checkpoint_path}.")

    def load_text_features(self, text_features_path: str, device=None):
        if device is None:
            device = next(self.parameters()).device
        checkpoint = torch.load(text_features_path, map_location="cpu")
        text_features = checkpoint['text_features'] if 'text_features' in checkpoint else checkpoint
        self.register_buffer('text_features', text_features.to(device))
        print(f"Text features caricate da: {text_features_path} (shape: {self.text_features.shape})")
        expected_classes = sum(self.num_classes)
        if self.text_features.shape[0] != expected_classes:
            print(f"Warning: Numero di classi nelle text features ({self.text_features.shape[0]}) "
                  f"diverso da quello atteso ({expected_classes})")

    def get_image_features(self, image, normalize: bool = False):
        x = self.visual(image)
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image: Optional[torch.Tensor] = None):
        if self.text_features.numel() == 0:
            raise RuntimeError("Text features non caricate!")
        if image is None:
            raise ValueError("image non può essere None")

        # Assumi text_features già L2-normalizzate; in caso contrario:
        # self.text_features = F.normalize(self.text_features, dim=-1)

        scale = self.logit_scale.exp()
        # split fisso in (age=9, gender=2, emotion=7) nell'ordine richiesto
        age_c, gender_c, emotion_c = self.num_classes
        text_age, text_gender, text_emotion = torch.split(self.text_features, (age_c, gender_c, emotion_c), dim=0)

        # ----- Caso NO VPT (num_prompt == 0): un solo pass + split -----
        if self.num_prompt <= 0:
            image_features = self.get_image_features(image, normalize=True)          # (B, D)
            logits_full   = scale * (image_features @ self.text_features.T)          # (B, 18)
            logits_age, logits_gender, logits_emotion = torch.split(logits_full, (age_c, gender_c, emotion_c), dim=1)
            return logits_age, logits_gender, logits_emotion

        # ----- Caso VPT -----
        if len(self.vpt_list) == 1:
            # Un solo VisionPromptLearner
            self.visual.prompt_learner = self.vpt_list[0]
            image_features = self.get_image_features(image, normalize=True)          # (B, D)
            logits_full   = scale * (image_features @ self.text_features.T)          # (B, 18)
            logits_age, logits_gender, logits_emotion = torch.split(logits_full, (age_c, gender_c, emotion_c), dim=1)
            return logits_age, logits_gender, logits_emotion
        else:
            # Più VPT: 3 forward nell'ordine age, gender, emotion
            # (se vuoi essere rigido, sblocca l'assert qui sotto)
            # assert len(self.vpt_list) >= 3, "Con VPT multipli servono almeno 3 prompt learner (age, gender, emotion)."

            # AGE
            self.visual.prompt_learner = self.vpt_list[0]
            img_feat_age  = self.get_image_features(image, normalize=True)           # (B, D)
            logits_age    = scale * (img_feat_age @ text_age.T)                      # (B, 9)

            # GENDER
            self.visual.prompt_learner = self.vpt_list[1]
            img_feat_gender = self.get_image_features(image, normalize=True)         # (B, D)
            logits_gender   = scale * (img_feat_gender @ text_gender.T)              # (B, 2)

            # EMOTION
            self.visual.prompt_learner = self.vpt_list[2]
            img_feat_emotion = self.get_image_features(image, normalize=True)        # (B, D)
            logits_emotion   = scale * (img_feat_emotion @ text_emotion.T)           # (B, 7)

            return logits_age, logits_gender, logits_emotion

    

    def load_pretrained_vision_encoder(self, name: str = "PE-Core-B16-224", checkpoint_path: str = None):
        from core.vision_encoder.config import fetch_pe_checkpoint
        ckpt_path = fetch_pe_checkpoint(name, checkpoint_path)
        self.visual.load_ckpt(ckpt_path)
        print(f"Vision encoder pretrained caricato da {ckpt_path}")
        
    @classmethod
    def from_inference_files(cls, vision_cfg, checkpoint_path: str, text_features_path: str,
                             vpt_checkpoint_paths=None, num_prompt=0, num_classes=(9,2,7), device='cpu'):
        num_vpt = len(vpt_checkpoint_paths) if vpt_checkpoint_paths is not None else 1
        model = cls(vision_cfg, num_prompt=num_prompt, num_classes=num_classes, num_vpt=num_vpt)
        model = model.to(device)
        model.load_from_checkpoint(checkpoint_path)
        if vpt_checkpoint_paths is not None:
            model.load_vpt_checkpoints(vpt_checkpoint_paths)
        model.load_text_features(text_features_path, device)
        model.eval()
        return model