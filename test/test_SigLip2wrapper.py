import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from wrappers.SigLip2.SigLip2Model import Siglip2Model
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor
from transformers import AutoModel
from PIL import Image
import torch
import pandas as pd


if __name__ == "__main__":
    # Example usage
    print("Starting SiglipModel test...")

    
    model_name = "google/siglip2-large-patch16-384"
    num_prompts = 0
    siglip2Config = AutoConfig.from_pretrained(model_name, cache_dir="./hf_models")
    custom_model = Siglip2Model(siglip2Config, num_prompts=num_prompts)
    custom_model.load_model(
        path="./hf_models/model.pt",
        repo_id=model_name,
        map_location="cpu"
    )

    model = AutoModel.from_pretrained(model_name, cache_dir="./hf_models")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./hf_models")
    image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="./hf_models")
    

    print("#" * 39)
    print("###                         Configuration                        ###")
    print("#" * 39)
    print("#"*10 + "Custom Model Configuration " + "#"*10)
    print(custom_model.config)
    print("#"*10 + " End Custom Model Configuration " + "#"*10)
    print("\n\n\n")
    print("#"*10 + "Model Configuration " + "#"*10)
    print(model.config)
    print("#"*10 + " End Model Configuration " + "#"*10)

    print("Equal config:", custom_model.config == model.config)
    if custom_model.config != model.config:
        print("Differences in configuration:")
        for key in custom_model.config.to_dict():
            if custom_model.config.to_dict().get(key) != model.config.to_dict().get(key):
                print(f"  {key}: Custom Model -> {custom_model.config.to_dict().get(key)}, Model -> {model.config.to_dict().get(key)}")    


    # Testing the forward pass and the model outputs
    image = Image.open("./test/image.jpg")
    text = ["A photo of a male", "A photo of a female",
            "A photo of a person between 0 and 2 years old",
            "A photo of a person between 3 and 9 years old",
            "A photo of a person between 10 and 19 years old",
            "A photo of a person between 20 and 29 years old",
            "A photo of a person between 30 and 39 years old",
            "A photo of a person between 40 and 49 years old",
            "A photo of a person between 50 and 59 years old",
            "A photo of a person between 60 and 69 years old",
            "A photo of a person with more than 70 years old",
            "A photo of a surprised person",
            "A photo of a fearful person",
            "A photo of a disgusted person",
            "A photo of a happy person",
            "A photo of a sad person",
            "A photo of an angry person",
            "A photo of a neutral person"
        ]
    prompt_prefix = " " + " ".join(["X"] * 16)
    prompts = [prompt_prefix + " " + name + "." for name in text]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model.to(device)
    model.to(device)

    # Separate processing for text and image
    text_inputs = tokenizer(prompts, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    print(tokenizer(tokenizer.eos_token,return_tensors="pt", padding='max_length', max_length=64, truncation=True))
    print(text_inputs['input_ids'][0])

    image_inputs = image_processor(images=image, return_tensors="pt")
    print("Tokenizer type:", tokenizer.__class__.__name__)
    print("Image processor type:", image_processor.__class__.__name__)
    print("Image processor main information (JSON):")
    try:
        print(image_processor.to_json_string())
    except AttributeError:
        print("  JSON export not available for this image processor.")


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
        forward = custom_model(text=text_inputs["input_ids"].to(device), image=image_inputs["pixel_values"].to(device))

    print("Outputs logit_per_image are equal:", torch.allclose(outputs.logits_per_image, custom_image_logit, atol=1e-9))
    print("Outputs logit_per_text are equal:", torch.allclose(outputs.logits_per_text, custom_text_logit, atol=1e-9))
    print("Outputs text_embeds are equal:", torch.allclose(outputs.text_embeds, custom_text_emb, atol=1e-9))
    print("Outputs image_embeds are equal:", torch.allclose(outputs.image_embeds, custom_image_emb, atol=1e-9))

    print("Forward pass completed successfully.")

    # Separate logits into tasks for both models
    gender_logits_custom = custom_image_logit[:, :2]
    age_logits_custom = custom_image_logit[:, 2:11]
    emotion_logits_custom = custom_image_logit[:, 11:]

    gender_logits_normal = outputs.logits_per_image[:, :2]
    age_logits_normal = outputs.logits_per_image[:, 2:11]
    emotion_logits_normal = outputs.logits_per_image[:, 11:]

    # Apply softmax to each task for both models
    gender_probs_custom = gender_logits_custom.softmax(dim=-1)
    age_probs_custom = age_logits_custom.softmax(dim=-1)
    emotion_probs_custom = emotion_logits_custom.softmax(dim=-1)

    gender_probs_normal = gender_logits_normal.softmax(dim=-1)
    age_probs_normal = age_logits_normal.softmax(dim=-1)
    emotion_probs_normal = emotion_logits_normal.softmax(dim=-1)

    # Initialize additional custom models
    custom_model_10_prompts = Siglip2Model(siglip2Config, num_prompts=10)
    custom_model_10_prompts.load_model(
        path="./hf_models/model.pth",
        map_location="cpu"
    )
    custom_model_30_prompts = Siglip2Model(siglip2Config, num_prompts=30)
    custom_model_30_prompts.load_model(
        path="./hf_models/model.pth",
        map_location="cpu"
    )

    # Move additional models to device
    custom_model_10_prompts.to(device)
    custom_model_30_prompts.to(device)

    # Perform inference with additional models
    with torch.no_grad():
        custom_image_emb_10 = custom_model_10_prompts.get_image_features(image_inputs["pixel_values"].to(device), normalize=True)
        custom_text_emb_10 = custom_model_10_prompts.get_text_features(text_inputs["input_ids"].to(device), normalize=True)
        custom_image_logit_10 = custom_model_10_prompts.logit_scale.exp() * (custom_image_emb_10 @ custom_text_emb_10.T) + custom_model_10_prompts.logit_bias

        custom_image_emb_30 = custom_model_30_prompts.get_image_features(image_inputs["pixel_values"].to(device), normalize=True)
        custom_text_emb_30 = custom_model_30_prompts.get_text_features(text_inputs["input_ids"].to(device), normalize=True)
        custom_image_logit_30 = custom_model_30_prompts.logit_scale.exp() * (custom_image_emb_30 @ custom_text_emb_30.T) + custom_model_30_prompts.logit_bias

    # Separate logits into tasks for additional models
    gender_logits_10 = custom_image_logit_10[:, :2]
    age_logits_10 = custom_image_logit_10[:, 2:11]
    emotion_logits_10 = custom_image_logit_10[:, 11:]

    gender_logits_30 = custom_image_logit_30[:, :2]
    age_logits_30 = custom_image_logit_30[:, 2:11]
    emotion_logits_30 = custom_image_logit_30[:, 11:]

    # Apply softmax to each task for additional models
    gender_probs_10 = gender_logits_10.softmax(dim=-1)
    age_probs_10 = age_logits_10.softmax(dim=-1)
    emotion_probs_10 = emotion_logits_10.softmax(dim=-1)

    gender_probs_30 = gender_logits_30.softmax(dim=-1)
    age_probs_30 = age_logits_30.softmax(dim=-1)
    emotion_probs_30 = emotion_logits_30.softmax(dim=-1)

    # Create a table with probabilities for all models
    data = {
        "Class": text[:2] + text[2:11] + text[11:],
        "Task": ["Gender"] * 2 + ["Age"] * 9 + ["Emotion"] * 7,
        "Custom Model (0 Prompts)": gender_probs_custom[0].tolist() + age_probs_custom[0].tolist() + emotion_probs_custom[0].tolist(),
        "Custom Model (10 Prompts)": gender_probs_10[0].tolist() + age_probs_10[0].tolist() + emotion_probs_10[0].tolist(),
        "Custom Model (30 Prompts)": gender_probs_30[0].tolist() + age_probs_30[0].tolist() + emotion_probs_30[0].tolist(),
        "Normal Model Probability": gender_probs_normal[0].tolist() + age_probs_normal[0].tolist() + emotion_probs_normal[0].tolist()
    }
    df = pd.DataFrame(data)
    print("\nComparison of probabilities by task:")
    print(df)
