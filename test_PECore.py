import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from wrappers.PerceptionEncoder.pe import PECore

if __name__ == '__main__':
    print("Starting PE-Core test...")
    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
    custom_model = PECore.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\n")
    print("#"*10 + "Custom Model Configuration " + "#"*10)
    print(f"Model type: {type(custom_model)}")
    print(f"Vision encoder: {type(custom_model.visual)}")
    print("#"*10 + " End Custom Model Configuration " + "#"*10)
    print("\n\n")
    print("#"*10 + "Model Configuration " + "#"*10)
    print(f"Model type: {type(model)}")
    print(f"Vision encoder: {type(model.visual)}")
    print("#"*10 + " End Model Configuration " + "#"*10)
    print("\n\n")
    print("Testing forward pass...")


    model = model.to(device)
    custom_model = custom_model.to(device)

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    pil_images = Image.open("./test/image.jpg")
    image = preprocess(pil_images).unsqueeze(0).to(device)
    text = tokenizer(["A photo of a male", "A photo of a female"]).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, text)
        text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        custom_image_features, custom_text_features, custom_logit_scale = custom_model(image, text)
        custom_text_probs = (custom_logit_scale * custom_image_features @ custom_text_features.T).softmax(dim=-1)

    print("Outputs image_features are equal:", torch.allclose(image_features, custom_image_features, atol=1e-9))
    print("Outputs text_features are equal:", torch.allclose(text_features, custom_text_features, atol=1e-9))
    print("Outputs logit_scale is equal:", torch.allclose(logit_scale, custom_logit_scale, atol=1e-9))
    print("Text probabilities are equal:", torch.allclose(text_probs, custom_text_probs, atol=1e-9))
    print("Forward pass completed successfully.")
    