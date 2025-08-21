import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from wrappers.SigLip2.SigLip2Model import Siglip2Model
from transformers import AutoTokenizer, AutoImageProcessor
from transformers import AutoModel
from PIL import Image
import torch


if __name__ == "__main__":
    # Example usage
    print("Starting SiglipModel test...")
    custom_model = Siglip2Model.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    
    tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    
    print("#"*10 + "Custom Model Configuration " + "#"*10)
    print(custom_model.config)
    print("#"*10 + " End Custom Model Configuration " + "#"*10)

    print("#"*10 + "Model Configuration " + "#"*10)
    print(model.config)
    print("#"*10 + " End Model Configuration " + "#"*10)

    print("Equal config:", custom_model.config == model.config)

    print("Testing forward pass...")

    image = Image.open("./test/image.jpg")
    text = ["A photo of a male", "A photo of a female"]
    prompt_prefix = " " + " ".join(["X"] * 16)
    prompts = [prompt_prefix + " " + name + "." for name in text]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model.to(device)
    model.to(device)

    # Separate processing for text and image
    text_inputs = tokenizer(prompts, return_tensors="pt", padding=True, max_length=64)
    image_inputs = image_processor(images=image, return_tensors="pt")
    print("Tokenizer type:", tokenizer.__class__.__name__)
    print("Image processor type:", image_processor.__class__.__name__)
    print("Tokenizer init kwargs:")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for k, v in init_kwargs.items():
            print(f"  {k}: {v}")
    else:
        print("  Not available")
    print("Image processor init kwargs:")
    img_kwargs = getattr(image_processor, "init_kwargs", None)
    if isinstance(img_kwargs, dict):
        for k, v in img_kwargs.items():
            print(f"  {k}: {v}")
    else:
        print("  Not available")
    # Stampa le trasformazioni dell'image processor
    print("Image processor transforms:")
    transforms = getattr(image_processor, "transform", None)
    if transforms is not None:
        print(f"  {transforms}")
    elif hasattr(image_processor, "image_processor") and hasattr(image_processor.image_processor, "transform"):
        print(f"  {image_processor.image_processor.transform}")
    else:
        print("  Not available")
    # Merge inputs
    inputs = {**text_inputs, **image_inputs}
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Inputs prepared")
    print("Inputs:")
    print(f"IDS: {inputs['input_ids'].shape}")
    print(f"Inputs_ids: {inputs['input_ids']}")
    print(f"PIXEL_VALUES: {inputs['pixel_values'].shape}")
    print("\n\n\n\n")


    
    with torch.no_grad():        
        outputs = model(**inputs, return_loss=True)

        custom_text_emb = custom_model.get_text_features(text_inputs["input_ids"].to(device), normalize=True)
        custom_image_emb = custom_model.get_image_features(image_inputs["pixel_values"].to(device), normalize=True)
        custom_image_logit = custom_model.logit_scale.exp() * (custom_image_emb @ custom_text_emb.T) + custom_model.logit_bias
        custom_text_logit = custom_image_logit.T           

    print("Outputs logit_per_image are equal:", torch.allclose(outputs.logits_per_image, custom_image_logit, atol=1e-9))
    print("Outputs logit_per_text are equal:", torch.allclose(outputs.logits_per_text, custom_text_logit, atol=1e-9))
    print("Outputs text_embeds are equal:", torch.allclose(outputs.text_embeds, custom_text_emb, atol=1e-9))
    print("Outputs image_embeds are equal:", torch.allclose(outputs.image_embeds, custom_image_emb, atol=1e-9))

    print("Forward pass completed successfully.")

     # Calcolo e stampa delle probabilità delle due classi per entrambi i modelli
    probs_custom = outputs.logits_per_image.softmax(dim=-1)
    probs_original = custom_image_logit.softmax(dim=-1)

    print("\nProbabilità classi (Custom Model):")
    for idx, label in enumerate(text):
        print(f"  {label}: {probs_custom[0, idx].item():.4f}")

    print("\nProbabilità classi (Original Model):")
    for idx, label in enumerate(text):
        print(f"  {label}: {probs_original[0, idx].item():.4f}")
