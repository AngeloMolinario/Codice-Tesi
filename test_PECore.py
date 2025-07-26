import torch
import json
import os
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from wrappers.PerceptionEncoder.pe import PECore

# === CONFIG ===
TEST_PROMPTS = ["A photo of a male", "A photo of a female"]
IMAGE_PATH = "./test/image.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === UTILITY ===
def check_all_equal(tensors, name, result_log):
    base = tensors[0]
    equal = all(torch.allclose(base, t, atol=1e-9) for t in tensors[1:])
    result_log[name] = bool(equal)
    print(f"{name}: {'TRUE' if equal else 'FALSE'}")
    return equal

def print_probabilities(prob_log_section, label_type):
    print(f"\n=== PROBABILITIES ({label_type.upper()}) ===")
    for model_name, probs in prob_log_section.items():
        print(f"{model_name}:")
        for label, prob in probs.items():
            print(f"  {label}: {prob:.4f}")

def print_model_info(models, names):
    for model, name in zip(models, names):
        print(f"\n{'#'*10} {name} Configuration {'#'*10}")
        print(f"Model type: {type(model)}")
        print(f"Vision encoder: {type(model.visual)}")
        print(f"{'#'*10} End {name} Configuration {'#'*10}\n")

# === TEST FUNCTIONS ===
def test_embeddings(models, names, image, text, normalize=False, result_log=None, prob_log=None):
    results = []
    for model in models:
        with torch.no_grad():
            image_features = model.encode_image(image, normalize=normalize)
            text_features = model.encode_text(text, normalize=normalize)
            logit_scale = model.logit_scale.exp()
            probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
            results.append((image_features, text_features, logit_scale, probs))

    suffix = "normalized" if normalize else "unnormalized"
    print(f"\n=== {suffix.upper()} ===")
    check_all_equal([r[0] for r in results], f"image_features_{suffix}", result_log)
    check_all_equal([r[1] for r in results], f"text_features_{suffix}", result_log)
    check_all_equal([r[2] for r in results], f"logit_scale_{suffix}", result_log)
    check_all_equal([r[3] for r in results], f"class_probabilities_{suffix}", result_log)

    prob_log[suffix] = {
        name: {
            label: float(probs[0][i])
            for i, label in enumerate(TEST_PROMPTS)
        }
        for name, (_, _, _, probs) in zip(names, results)
    }

    print_probabilities(prob_log[suffix], suffix)

def test_forward_pass(models, names, image, text, result_log=None, prob_log=None):
    results = []
    for model in models:
        with torch.no_grad():
            image_features, text_features, logit_scale = model(image, text)
            probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
            results.append((image_features, text_features, logit_scale, probs))

    print(f"\n=== FORWARD PASS ===")
    check_all_equal([r[0] for r in results], "image_features_forward", result_log)
    check_all_equal([r[1] for r in results], "text_features_forward", result_log)
    check_all_equal([r[2] for r in results], "logit_scale_forward", result_log)
    check_all_equal([r[3] for r in results], "class_probabilities_forward", result_log)

    prob_log["forward"] = {
        name: {
            label: float(probs[0][i])
            for i, label in enumerate(TEST_PROMPTS)
        }
        for name, (_, _, _, probs) in zip(names, results)
    }

    print_probabilities(prob_log["forward"], "forward")

# === MAIN ===
def main():
    model_configs = [
        ("PE-Core-B16-224", {"num_prompt": None}),
        ("Custom Model (No prompts)", {"num_prompt": 0}),
        ("Custom Model (1 prompt)", {"num_prompt": 1}),
        ("Custom Model (10 prompts)", {"num_prompt": 10}),
        ("Custom Model (50 prompts)", {"num_prompt": 50}),
        ("Custom Model (100 prompts)", {"num_prompt": 100}),
    ]

    models = []
    for name, kwargs in model_configs:
        print(f"Loading model: {name} with config: {kwargs}")
        if name == "PE-Core-B16-224":
            models.append(pe.CLIP.from_config(name, pretrained=True).to(DEVICE).eval())
        else:
            models.append(PECore.from_config("PE-Core-B16-224", pretrained=True, **kwargs).to(DEVICE).eval())

    model_names = [name for name, _ in model_configs]
    print_model_info(models, model_names)

    preprocess = transforms.get_image_transform(models[0].image_size)
    tokenizer = transforms.get_text_tokenizer(models[0].context_length)

    image = preprocess(Image.open(IMAGE_PATH)).unsqueeze(0).to(DEVICE)
    text = tokenizer(TEST_PROMPTS).to(DEVICE)

    results = {}
    probabilities = {}

    test_embeddings(models, model_names, image, text, normalize=False, result_log=results, prob_log=probabilities)
    test_embeddings(models, model_names, image, text, normalize=True, result_log=results, prob_log=probabilities)
    test_forward_pass(models, model_names, image, text, result_log=results, prob_log=probabilities)

    os.makedirs("test_result", exist_ok=True)

    with open("test_result/test_PECore.json", "w") as f:
        json.dump({
            "comparison_results": results,
            "classification_probabilities": probabilities
        }, f, indent=4)

    print("\nResults saved to results.json")

if __name__ == '__main__':
    main()
