import torch
from torchvision.transforms import transforms as T
import os

from transformers import Autoconfig

from core.vision_encoder import transforms
from wrappers.PerceptionEncoder.pe import PECore
from wrappers.promptopt.prompt_learner import CustomModel
from wrappers.SigLip2.SigLip2Model import Siglip2Model
from wrappers.tokenizer import PETokenizer, SigLip2Tokenizer

def get_image_transform(config):
    '''
        Get the image transformation used during the pretraining of the specific model.
    '''
    model_name = config.MODEL.lower()
    if model_name == 'pecore':
        return transforms.get_image_transform(224)
    elif model_name == 'siglip2':
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    raise ValueError(f"Unknown model name: {model_name}")

def get_tokenizer(config):
    ''' Get the tokenizer for the specific model.
        The tokenizer used are wrapped in a specific class so to use the correct one for each model with the correct initialization parameters
    '''
    if config.MODEL.lower() == 'pecore':
        return PETokenizer.get_instance(32) # The PECore model used has a context length of 32
    elif config.MODEL.lower() == 'siglip2':
        return SigLip2Tokenizer.get_instance(64) # The Siglip2 model used has a contenxt length of 64


def get_model(config):
    ''' This method look at the configuration file and return the correct model initialized with pretrained weights and the specified attributes'''
    tuning = config.TUNING.lower()
    model_name = config.MODEL.lower()

    if tuning == "softcpt":
        if model_name == "pecore":
            base_model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            model = CustomModel(
                n_ctx=config.NUM_TEXT_CNTX,
                tasknames=config.TASK_NAMES,
                classnames=config.CLASSES,
                model=base_model,
                tokenizer=transforms.get_text_tokenizer(base_model.text_model.context_length)
            )
            return model
        
        elif model_name == "siglip2":
            base_model = Siglip2Model(
                config=AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
                num_prompts=config.NUM_VISUAL_PROMPT
            )
            base_model.load_model(path="./hf_models/model.pth", map_location="cpu")
            model = CustomModel(
                n_ctx=config.NUM_TEXT_CNTX,
                tasknames=config.TASK_NAMES,
                classnames=config.CLASSES,
                model=base_model,
                tokenizer=get_tokenizer(config)
            )
            return model
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    elif tuning == "vpt":
        if model_name == "pecore":
            model = PECore.from_config("PE-Core-B16-224", pretrained=True, num_prompt=config.NUM_VISUAL_PROMPT)
            return model
        
        elif model_name == "siglip2":
            model = Siglip2Model(
                config = AutoConfig.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models"),
                num_prompts=config.NUM_VISUAL_PROMPT
            )
            model.load_model(path="./hf_models/model.pth", map_location="cpu")
            return model
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    elif tuning == "vvpt":
        raise NotImplementedError(f"Model {model_name} is not implemented for VVPT tuning.")
    else:
        raise ValueError(f"Unknown tuning method: {tuning}")