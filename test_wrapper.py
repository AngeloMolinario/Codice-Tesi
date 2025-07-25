from wrappers.SigLip2Model import SiglipModel
from transformers import AutoProcessor
from transformers import AutoModel
from PIL import Image
import torch


if __name__ == "__main__":
    # Example usage
    print("Starting SiglipModel test...")
    custom_model = SiglipModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", cache_dir="./hf_models")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model.to(device)
    model.to(device)

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, max_length=64, truncation=True).to(device)


    with torch.no_grad():
        outputs = custom_model(**inputs)
        custom_output = model(**inputs)

    print("Outputs logit_per_image are equal:", torch.allclose(outputs.logits_per_image, custom_output.logits_per_image, atol=1e-9))
    print("Outputs logit_per_text are equal:", torch.allclose(outputs.logits_per_text, custom_output.logits_per_text, atol=1e-9))
    print("Outputs text_embeds are equal:", torch.allclose(outputs.text_embeds, custom_output.text_embeds, atol=1e-9))
    print("Outputs image_embeds are equal:", torch.allclose(outputs.image_embeds, custom_output.image_embeds, atol=1e-9))
    if outputs.loss is None:
        print("Outputs loss is None in custom model, checking if it's None in the original model...")
    if custom_output.loss is None:
        print("Custom output loss is None, checking if it's None in the outputs...")
    if outputs.loss is not None and custom_output.loss is not None:
        print("Outputs loss is equal:", torch.allclose(outputs.loss, custom_output.loss, atol=1e-9))
    
    print("Outputs are equal:", outputs == custom_output)
    print("Forward pass completed successfully.")
