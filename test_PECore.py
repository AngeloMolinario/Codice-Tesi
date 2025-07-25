import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


if __name__ == '__main__':
    print("Starting PE-Core test...")
    print("CLIP configs:", pe.CLIP.available_configs())
    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    pil_images = Image.open("./test/image.jpg")
    image = preprocess(pil_images).unsqueeze(0).cuda()
    text = ["A photo of a male", "A photo of a female"].cuda

    with torch.no_grad(), torch.autocast("cuda"):
        image_features, text_features, logit_scale = model(image, text)
        text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]