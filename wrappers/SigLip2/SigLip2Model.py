from typing import Optional
import torch
from torch import nn
from transformers import AutoConfig
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
            pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)

        return pooled_output

    def get_image_features(self, image, normalize=False):
        '''Return the final pooled image features, if normalize is True, the output will be normalized.'''

        vision_outputs = self.vision_model(
            pixel_values=image,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )
        pooled_output = vision_outputs.pooler_output
        if normalize:
            pooled_output = pooled_output / pooled_output.norm(p=2, dim=-1, keepdim=True)
            
        return pooled_output

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




if __name__ == "__main__":
    # Example usage
    model = Siglip2Model.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    print("#"*10 + " Model Configuration " + "#"*10)
    print(model.config)
    print("#"*10 + " Model Summary " + "#"*10)
