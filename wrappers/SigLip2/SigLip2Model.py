from typing import Optional
import torch
from torch import nn
from transformers import AutoConfig

from wrappers.promptopt.prompt_learner import VisionPromptLearner
from .vision import CustomSiglipVisionTransformer, SiglipVisionModel
from .text import SiglipTextModel
import os

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


from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


def download_from_hub(repo_id, filename, cache_dir=None):
    print("Downloading from Hugging Face Hub...")
    local_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    print(f"Downloaded to {local_file}")
    return local_file

def safetensors_to_pth(safetensors_file, output_dir):
    # Load the safetensors file
    tensors = load_file(safetensors_file)
    model = Siglip2Model(AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir=output_dir), 0)
    model.load_state_dict(tensors)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))


class Siglip2Model(SiglipPreTrainedModel):

    config: SiglipConfig

    def __init__(self, config: SiglipConfig, num_prompts: int = 0):
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
        

        # Added for compatibility with the HF default configuration
        config.torch_dtype = "float32"
        config.text_config.torch_dtype = "float32"
        config.vision_config.torch_dtype = "float32"
        self.config = config    
        text_config = config.text_config
        vision_config = config.vision_config
        # Initialize the text and vision models with the number of prompts
        self.text_model = SiglipTextModel(text_config)
        # If num prompt is 0 than the model is the pure baseline
        self.vision_model = SiglipVisionModel(vision_config, num_prompt=num_prompts)


        # Added to maintain the backward compatibility for the weights loading
        self.text_model = self.text_model.text_model
        self.vision_model = self.vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()    

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

        torch.save(self.state_dict(), save_path)
        print(f"Salvati parametri vision encoder e logit_scale in {save_path}")

        # Salva opzionalmente le text features
        if text_features is not None and text_features_path is not None:
            os.makedirs(os.path.dirname(text_features_path), exist_ok=True)
            torch.save({'text_features': text_features.cpu()}, text_features_path)
            print(f"Text features salvate in {text_features_path} (shape: {text_features.shape})")

    def save_vision_model(self, output_dir: str, filename: str = "vision_ckpt.pt"):
        """
        Save vision-only checkpoint with full-model key format for Siglip2Model.
        - Prefix keys with `vision_model.` to match full model state_dict format.
        - Exclude VPT params (prompt_learner.*).
        - Include `logit_scale` and (if present) `logit_bias`.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        visual_sd = self.vision_model.state_dict()
        out_sd = {}
        for k, v in visual_sd.items():
            if k.startswith("prompt_learner"):
                continue
            out_sd[f"vision_model.{k}"] = v.detach().cpu()
        # Add scale/bias
        out_sd["logit_scale"] = self.logit_scale.detach().cpu()
        if hasattr(self, "logit_bias") and isinstance(self.logit_bias, torch.nn.Parameter):
            out_sd["logit_bias"] = self.logit_bias.detach().cpu()
        save_path = os.path.join(output_dir, filename)
        torch.save(out_sd, save_path)
        print(f"[Siglip2Model] Vision model saved (vision_model.* + logit_scale[/bias]) to {save_path}")

    def save_logit(self, save_path, filename="logits.pt"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = {
            'logit_scale' : self.logit_scale,
            'logit_bias' : self.logit_bias
        }
        torch.save(state_dict, os.path.join(save_path, filename))
        print(f"Logit scale and bias saved in {os.path.join(save_path, filename)}")

    def save_vpt_token(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if hasattr(self.vision_model, 'prompt_learner') and self.vision_model.prompt_learner is not None:
            state_dict = {
                'prompt_learner' : self.vision_model.prompt_learner.state_dict()
            }
            torch.save(state_dict, save_path)
            print(f"VPT token saved in {save_path}")

        else:
            print(f"Prompt learner not found in vision model, skipping save.")
                
    def load_model(self, path, map_location, repo_id="google/siglip2-base-patch16-224", filename="model.safetensors"):
        # Load the model weights from a local path, if the model is not found than it is downloaded from the hub, saved in the given path and loaded

        # Check if the file exists
        if not os.path.exists(path):
            print("Model not found. Downloading it from the hub...")
            # If not, download it from the hub
            sf_path = download_from_hub(repo_id, filename, cache_dir=os.path.dirname(path))
            # Convert the model to .pth format
            safetensors_to_pth(sf_path, output_dir=os.path.dirname(path))
        # Load the model
        state_dict = torch.load(os.path.join(os.path.dirname(path), "model.pth"), map_location=map_location)
        result = self.load_state_dict(state_dict, strict=False)
        
        # Check for missing or mismatching keys during the loading of the state_dict
        if result.missing_keys:
            print("[WARNING] The loaded models miss the following keys, ensure to train it on a downstream task:")
            for k in result.missing_keys:
                print(f"  {k}")
        if result.unexpected_keys:
            print("[WARNING] The loaded models have unexpected keys:")
            for k in result.unexpected_keys:
                print(f"  {k}")

        
        print(f"LOGIT SCALE: {self.logit_scale} - exp {self.logit_scale.exp()}")
        print(f"LOGIT BIAS: {self.logit_bias}")

        return result

    def get_text_features(self, text, normalize=False):
        '''Return the final pooled text features, if normalize is True, the output will be normalized.'''

        text_outputs = self.text_model(
            input_ids=text,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )

        pooled_output = text_outputs.pooler_output

        if normalize:
            #pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)
            pooled_output = nn.functional.normalize(pooled_output, dim=-1) if normalize else pooled_output
        return pooled_output

    def get_image_features(self, image, normalize=False):
        '''Return the final pooled image features, if normalize is True, the output will be normalized.'''

        vision_outputs = self.vision_model(
            pixel_values=image,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )
        if normalize:
            #pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)
            vision_outputs = nn.functional.normalize(vision_outputs, dim=-1) if normalize else vision_outputs
        return vision_outputs

    def forward(self, text, image):
        '''Return the cosine similarity between text and image features.
        Text and image features are obtained from their respective models, normalized and than their cosine similarity is scaled by the logit_Scale factor and a bias is added.
        The returned tensor has shape (B, num_image, text).
        '''

        # normalized features
        image_embeds = self.get_image_features(image, normalize=True)
        text_embeds = self.get_text_features(text, normalize=True)

        # cosine similarity as logits
        logits_per_image = torch.matmul(image_embeds, text_embeds.t().to(image_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_image = logits_per_image * logit_scale.exp() + logit_bias


        return logits_per_image


class Siglip2Vision(nn.Module):

    config: SiglipConfig

    def __init__(self, config: SiglipConfig, num_prompt=0):
        super().__init__()
        vision_config = config.vision_config
        
        if not isinstance(vision_config, SiglipVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config)}."
            )
        

        # Added for compatibility with the HF default configuration
        config.torch_dtype = "float32"        
        config.vision_config.torch_dtype = "float32"
        self.config = config            
        self.num_prompt = num_prompt
        
        # If num prompt is 0 than the model is the pure baseline
        self.vision_model = SiglipVisionModel(vision_config, num_prompt=num_prompt)
        
        # Added to maintain the backward compatibility for the weights loading        
        self.vision_model = self.vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self.register_buffer("text_features", torch.empty(0))

        self._vpt = []

    def save_vision_model(self, output_dir: str, filename: str = "vision_ckpt.pt"):
        """
        Save vision-only checkpoint with full-model key format for Siglip2Vision.
        - Prefix keys with `vision_model.` to match state_dict format.
        - Exclude VPT params (prompt_learner.*).
        - Include `logit_scale` and (if present) `logit_bias`.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        visual_sd = self.vision_model.state_dict()
        out_sd = {}
        for k, v in visual_sd.items():
            if k.startswith("prompt_learner"):
                continue
            out_sd[f"vision_model.{k}"] = v.detach().cpu()
        out_sd["logit_scale"] = self.logit_scale.detach().cpu()
        if hasattr(self, "logit_bias") and isinstance(self.logit_bias, torch.nn.Parameter):
            out_sd["logit_bias"] = self.logit_bias.detach().cpu()
        save_path = os.path.join(output_dir, filename)
        torch.save(out_sd, save_path)
        print(f"[Siglip2Vision] Vision model saved (vision_model.* + logit_scale[/bias]) to {save_path}")

    def get_image_features(self, image, normalize=True):
        vision_outputs = self.vision_model(
            pixel_values=image,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )
        if normalize:
            vision_outputs = nn.functional.normalize(vision_outputs, dim=-1) if normalize else vision_outputs
        return vision_outputs

    def get_task_image_features(self, task_id, image, normalize=True):
        self.vision_model.prompt_learner = self._vpt[task_id].to(self.vision_model.device)
        vision_outputs = self.vision_model(
            pixel_values=image,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )

        if normalize:
            vision_outputs = nn.functional.normalize(vision_outputs, dim=-1) if normalize else vision_outputs
        return vision_outputs
    
    def forward(self, image):

        if len(self._vpt) <= 1:
            logit = self.logit_scale.exp() * self.get_image_features(image) @ self.text_features.t()
            return torch.split(logit, [9, 2, 7], dim=-1)
        
        text_features = torch.split(self.text_features, [9, 2, 7], dim=0)
        age_logit = self.logit_scale.exp() * self.get_task_image_features(0, image) @ text_features[0].t()
        gender_logit = self.logit_scale.exp() * self.get_task_image_features(1, image) @ text_features[1].t()
        emotion_logit = self.logit_scale.exp() * self.get_task_image_features(2, image) @ text_features[2].t()
        
        return age_logit, gender_logit, emotion_logit

    def set_text_features(self, text_features):
        self.text_features = text_features

    def load_VPT_token(self, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'prompt_learner' in checkpoint:
            # Inizializza un'istanza di VisionPromptLearner
            vpt = VisionPromptLearner(
                emb_size=self.config.vision_config.hidden_size,
                num_prompt=self.num_prompt,
                is_cls_present=False
            )
            # Carica i pesi salvati nel prompt learner
            vpt.load_state_dict(checkpoint['prompt_learner'])
            # Aggiungi il prompt learner alla lista e spostalo sul dispositivo
            self._vpt.append(vpt.to(device))
            self.vision_model.prompt_learner = vpt.to(device)
            print(f"VPT token loaded from {ckpt_path}")
        else:
            print(f"No VPT token found in {ckpt_path}")

    def load_baseline(self, path, map_location, repo_id="google/siglip2-base-patch16-224", filename="model.safetensors"):
        """
        Load baseline weights for the vision backbone with broad compatibility:
        - If `path` exists and points to a checkpoint saved via `save_vision_model`, it expects keys
          prefixed with `vision_model.*` plus `logit_scale`/`logit_bias`.
        - Otherwise falls back to downloading/converting from the Hub and loading `model.pth`.
        """
        # Prefer explicit local checkpoint if present
        if path is not None and os.path.exists(path):
            ckpt = torch.load(path, map_location=map_location)
            # Unwrap nested dicts
            if isinstance(ckpt, dict) and ("state_dict" in ckpt or "weights" in ckpt):
                ckpt = ckpt.get("state_dict", ckpt.get("weights", ckpt))
            # Strip optional module. prefix
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

            # Load logit params if present
            if "logit_scale" in ckpt and hasattr(self, "logit_scale"):
                try:
                    self.logit_scale.data.copy_(ckpt["logit_scale"].to(map_location))
                except Exception:
                    pass
            if "logit_bias" in ckpt and hasattr(self, "logit_bias"):
                try:
                    self.logit_bias.data.copy_(ckpt["logit_bias"].to(map_location))
                except Exception:
                    pass

            # Extract only vision weights
            if any(k.startswith("vision_model.") for k in ckpt):
                vision_sd = {k.replace("vision_model.", ""): v for k, v in ckpt.items() if k.startswith("vision_model.")}
            else:
                # Assume it contains only vision weights
                vision_sd = {k: v for k, v in ckpt.items() if not k.startswith("text_model.")}

            # Exclude VPT if present
            vision_sd = {k: v for k, v in vision_sd.items() if not k.startswith("prompt_learner")}

            result = self.vision_model.load_state_dict(vision_sd, strict=False)
        else:
            # Fallback to Hub download/convert and load full state_dict
            print("Model not found locally. Downloading from the hub...")
            sf_path = download_from_hub(repo_id, filename, cache_dir=os.path.dirname(path) if path else None)
            safetensors_to_pth(sf_path, output_dir=os.path.dirname(path) if path else ".")
            state_dict = torch.load(os.path.join(os.path.dirname(path) if path else ".", "model.pth"), map_location=map_location)
            result = self.load_state_dict(state_dict, strict=False)

        # Report loading status
        if hasattr(result, 'missing_keys') and result.missing_keys:
            print("[WARNING] The loaded models miss the following keys, ensure to train it on a downstream task:")
            for k in result.missing_keys:
                if not k.startswith("text_model"):
                    print(f"  {k}")
        if hasattr(result, 'unexpected_keys') and result.unexpected_keys:
            print("[WARNING] The loaded models have unexpected keys:")
            for k in result.unexpected_keys:
                if not k.startswith("text_model"):
                    print(f"  {k}") 

        print(f"LOGIT SCALE: {self.logit_scale} - exp {self.logit_scale.exp()}")
        print(f"LOGIT BIAS: {self.logit_bias}")

        return result


if __name__ == "__main__":
    # Example usage
    model = Siglip2Model.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    print("#"*10 + " Model Configuration " + "#"*10)
    print(model.config)
    print("#"*10 + " Model Summary " + "#"*10)
