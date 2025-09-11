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
    
    
    def save_logit(self, output_dir: str, filename: str = "logits.pt"):
        """
        Save logit_scale and logit_bias (if present) to a checkpoint.
        """
        os.makedirs(output_dir, exist_ok=True)
        out_sd = {}
        out_sd["logit_scale"] = self.logit_scale.detach().cpu()
        save_path = os.path.join(output_dir, filename)
        torch.save(out_sd, save_path)
        print(f"[PECore] Logit parameters saved (logit_scale) to {save_path}")

    def save_vision_model(self, output_dir: str, filename: str = "vision_ckpt.pt"):
        """
        Save vision-only checkpoint in the same key format used by full-model loaders.
        - Keys are prefixed with `visual.` to match PE loaders.
        - Excludes VPT params (`prompt_learner.*`).
        - Also saves `logit_scale` and (if present) `logit_bias` at top level.
        """
        os.makedirs(output_dir, exist_ok=True)
        visual_sd = self.visual.state_dict()
        out_sd = {}
        for k, v in visual_sd.items():
            if k.startswith("prompt_learner"):
                continue
            out_sd[f"visual.{k}"] = v.detach().cpu()
        # Add scaling params
        out_sd["logit_scale"] = self.logit_scale.detach().cpu()
        if hasattr(self, "logit_bias") and isinstance(self.logit_bias, torch.nn.Parameter):
            out_sd["logit_bias"] = self.logit_bias.detach().cpu()
        save_path = os.path.join(output_dir, filename)
        torch.save(out_sd, save_path)
        print(f"[PECore] Vision model saved (visual.* + logit_scale[/bias]) to {save_path}")
    
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

    def save(self, save_path: str):                    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = {}
    
        # Salva tutti i parametri del vision encoder
        for name, param in self.visual.state_dict().items():
            state_dict[f'visual.{name}'] = param
    
        # Salva il logit_scale
        state_dict['logit_scale'] = self.logit_scale.detach().cpu()
    
        # Salva lo state_dict completo
        torch.save(state_dict, save_path)
        print(f"Tutti i parametri del modello salvati in {save_path}")

    def save_vpt_token(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if hasattr(self.visual, 'prompt_learner') and self.visual.prompt_learner is not None:
            state_dict = {
                'prompt_learner' : self.visual.prompt_learner.state_dict()
            }
            torch.save(state_dict, save_path)
            print(f"VPT token saved in {save_path}")
        else:
            print("Nessun prompt learner trovato nella componente visiva. Skipping save.")

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
    def __init__(self, vision_cfg, num_prompt):
        super().__init__()
        self.visual = CustomVisionTransformer(**asdict(vision_cfg), num_prompt=num_prompt)
        self.image_size = self.visual.image_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))        
        self.num_prompt = num_prompt
        self._vpt = []       
        self.register_buffer('text_features', torch.empty(0))
        self.logit_scale = nn.Parameter(torch.ones([]))

    def save_vision_model(self, output_dir: str, filename: str = "vision_ckpt.pt"):
        """
        Save vision-only checkpoint with PE full-model key format.
        - Prefix keys with `visual.`; exclude VPT params.
        - Include `logit_scale` and `logit_bias` if present.
        """
        os.makedirs(output_dir, exist_ok=True)
        visual_sd = self.visual.state_dict()
        out_sd = {}
        for k, v in visual_sd.items():
            if k.startswith("prompt_learner"):
                continue
            out_sd[f"visual.{k}"] = v.detach().cpu()
        out_sd["logit_scale"] = self.logit_scale.detach().cpu()
        if hasattr(self, "logit_bias") and isinstance(self.logit_bias, torch.nn.Parameter):
            out_sd["logit_bias"] = self.logit_bias.detach().cpu()
        save_path = os.path.join(output_dir, filename)
        torch.save(out_sd, save_path)
        print(f"[PECore_Vision] Vision model saved (visual.* + logit_scale[/bias]) to {save_path}")

    def get_image_features(self, x, normalize=True):
        features = self.visual(x)
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        
        return features
    
    def get_task_image_features(self, task_id, image, normalize=True):
        self.visual.prompt_learner = self._vpt[task_id]
        features = self.visual(image)
        if normalize:
            features = F.normalize(features, p=2, dim=-1)

        return features
    
    def forward(self, image):
        """
        Restituisce sempre una tupla (age_logit, gender_logit, emotion_logit).
        Regole:
          - Se text_features ha 18 vettori (9+2+7): multitask -> split o calcolo per-task con VPT multipli.
          - Se ha 9 vettori: solo age -> (age_logit, None, None)
          - Se ha 2 vettori: solo gender -> (None, gender_logit, None)
          - Se ha 7 vettori: solo emotion -> (None, None, emotion_logit)
        Funziona sia con nessun/un solo VPT (len(_vpt) <= 1) sia con VPT multipli (len(_vpt) > 1).
        """
        if self.text_features.numel() == 0:
            raise RuntimeError("Text features non impostate. Chiamare set_text_features prima della forward.")

        n_classes = self.text_features.shape[0]
        scale = self.logit_scale.exp()

        # -------------------- MULTITASK (18 = 9+2+7) --------------------
        if n_classes == 18:
            if len(self._vpt) <= 1:
                img_feat = self.get_image_features(image, normalize=True)
                logits = scale * (img_feat @ self.text_features.t())
                age_logit, gender_logit, emotion_logit = torch.split(logits, [9, 2, 7], dim=-1)
                return age_logit, gender_logit, emotion_logit
            else:
                age_tf, gender_tf, emotion_tf = torch.split(self.text_features, [9, 2, 7], dim=0)
                age_feat = self.get_task_image_features(0, image, normalize=True)
                gender_feat = self.get_task_image_features(1, image, normalize=True)
                emotion_feat = self.get_task_image_features(2, image, normalize=True)
                age_logit = scale * (age_feat @ age_tf.t())
                gender_logit = scale * (gender_feat @ gender_tf.t())
                emotion_logit = scale * (emotion_feat @ emotion_tf.t())
                return age_logit, gender_logit, emotion_logit

        # -------------------- SINGLE TASK (9 / 2 / 7) --------------------
        if n_classes == 9:  # AGE
            if len(self._vpt) >= 1:
                img_feat = self.get_task_image_features(0, image, normalize=True)
            else:
                img_feat = self.get_image_features(image, normalize=True)
            age_logit = scale * (img_feat @ self.text_features.t())
            return age_logit, None, None

        if n_classes == 2:  # GENDER
            if len(self._vpt) >= 1:
                img_feat = self.get_task_image_features(0, image, normalize=True)
            else:
                img_feat = self.get_image_features(image, normalize=True)
            gender_logit = scale * (img_feat @ self.text_features.t())
            return None, gender_logit, None

        if n_classes == 7:  # EMOTION
            if len(self._vpt) >= 1:
                img_feat = self.get_task_image_features(0, image, normalize=True)
            else:
                img_feat = self.get_image_features(image, normalize=True)
            emotion_logit = scale * (img_feat @ self.text_features.t())
            return None, None, emotion_logit

        raise ValueError(f"Numero di text features non supportato: {n_classes}. Attesi 9, 2, 7 oppure 18 (9+2+7).")

    def set_text_features(self, text_features):
        self.text_features = text_features
    
    def load_VPT_token(self, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'prompt_learner' in checkpoint:
            # Inizializza un'istanza di VisionPromptLearner
            vpt = VisionPromptLearner(
                emb_size=self.visual.width,
                num_prompt=self.num_prompt,
                is_cls_present=False
            )
            # Carica i pesi salvati nel prompt learner
            vpt.load_state_dict(checkpoint['prompt_learner'])
            # Aggiungi il prompt learner alla lista e spostalo sul dispositivo
            self._vpt.append(vpt.to(device))
            self.visual.prompt_learner = vpt.to(device)
            print(f"VPT token loaded from {ckpt_path}")
        else:
            print(f"No VPT token found in {ckpt_path}")

    def load_baseline(self, ckpt_path, device):
        """
        Carica il vision encoder da un checkpoint.
        Supporta sia i checkpoint originali PE che quelli salvati con `save_vision_model`.

        Args:
            ckpt_path (str): Percorso del checkpoint (può essere quello salvato con `save_vision_model`).
            device (torch.device): Dispositivo su cui caricare il modello.
        """
        # Scegli la sorgente: se esiste un file locale usa quello, altrimenti il checkpoint di default
        path = ckpt_path if (ckpt_path is not None and os.path.exists(ckpt_path)) else fetch_pe_checkpoint("PE-Core-B16-224")

        _sd = torch.load(path, map_location=device)
        if isinstance(_sd, dict) and ("state_dict" in _sd or "weights" in _sd):
            _sd = _sd.get("state_dict", _sd.get("weights", _sd))

        # compatibilità con chiavi prefissate con 'module.'
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}

        # carica logit_scale/logit_bias se presenti
        if "logit_scale" in _sd:
            self.logit_scale.data.copy_(_sd["logit_scale"].to(device))
        if hasattr(self, "logit_bias") and "logit_bias" in _sd:
            try:
                self.logit_bias.data.copy_(_sd["logit_bias"].to(device))
            except Exception:
                pass

        # Estrai solo i pesi del vision encoder; supporta formati con prefisso 'visual.' (save_vision_model)
        if any(k.startswith("visual.") for k in _sd):
            visual_sd = {k.replace("visual.", ""): v for k, v in _sd.items() if k.startswith("visual.")}
        else:
            # se non c'è prefisso, assume che lo sd contenga solo i pesi del vision encoder
            visual_sd = {k: v for k, v in _sd.items() if not k.startswith("text_model.")}

        # Escludi eventuali pesi di VPT
        visual_sd = {k: v for k, v in visual_sd.items() if not k.startswith("prompt_learner")}

        m, u = self.visual.load_state_dict(visual_sd, strict=False)

        if (m or u):
            print(f"Missing keys for loading vision encoder: {m}")
            print(f"Unexpected keys for loading vision encoder: {u}")

        print(f"logit scale {self.logit_scale.item()}")
