'''
This script compute the concordance index for two models on a given dataset.
The index is used to assess the effectiveness of one methodoly over another, by comparing 
their predictions against the baseline outcomes.

The baseline models are prompted using a standardized prompt template for classification tasks:

        "A photo of a {label}".

The index is computed by taking all the predictions from the baseline model and comparing them
to the predictions from the other model, counting the number of times where the predictions agree
with the baseline outcomes. The final concordance index is the ratio of the number of agreements
to the total number of comparisons made.
'''

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 

from core.vision_encoder.config import PE_VISION_CONFIG
from dataset.dataset import *
from core.vision_encoder.pe import CLIP
from core.vision_encoder.transforms import get_image_transform
from wrappers.PerceptionEncoder.pe import PECore_Vision
from wrappers.tokenizer import PETokenizer

CLASSES = [
    ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ["male", "female"],
    ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
]

TEXT_PROMPTS = [
    [
        "a photo of a person between 0 and 2 years old",
        "a photo of a person between 3 and 9 years old",
        "a photo of a person between 10 and 19 years old",
        "a photo of a person between 20 and 29 years old",
        "a photo of a person between 30 and 39 years old",
        "a photo of a person between 40 and 49 years old",
        "a photo of a person between 50 and 59 years old",
        "a photo of a person between 60 and 69 years old",
        "a photo of a person 70 years old or older"
    ],
    [
        "a photo of a male person",
        "a photo of a female person"
    ],
    [
        "a photo of a surprised person",
        "a photo of a fearful person",
        "a photo of a disgusted person",
        "a photo of a happy person",
        "a photo of a sad person",
        "a photo of an angry person",
        "a photo of a neutral person"
    ]
]

def format_table(headers, rows):
    """Format table without external dependencies."""
    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator line
    separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    
    # Format header
    header_row = "|" + "|".join(f" {header:<{col_widths[i]}} " for i, header in enumerate(headers)) + "|"
    
    # Format data rows
    formatted_rows = []
    for row in rows:
        formatted_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        formatted_rows.append(formatted_row)
    
    # Combine all parts
    table_lines = [separator, header_row, separator] + formatted_rows + [separator]
    return "\n".join(table_lines)

def discover_checkpoints(ckpt_dir):
    """Discover model checkpoints and features."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    parent_dir = os.path.dirname(ckpt_dir)
    
    # Vision checkpoint
    vision_ckpt = None
    for path in [
        os.path.join(ckpt_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "vision_ckpt.pt"),
        os.path.join(parent_dir, "full_training_model.pt")
    ]:
        if os.path.isfile(path):
            vision_ckpt = path
            break
    
    # VPT tokens
    vpt_tokens = []
    if os.path.isdir(ckpt_dir):
        for fn in sorted(os.listdir(ckpt_dir)):
            if fn.startswith("vpt_token") and fn.endswith(".pt"):
                vpt_tokens.append(os.path.join(ckpt_dir, fn))
    
    # Text features
    text_feats = None
    for path in [
        os.path.join(ckpt_dir, f"text_features_bval.pt"),
        os.path.join(ckpt_dir, "text_features.pt")
    ]:
        if os.path.isfile(path):
            text_feats = path
            break

    logit_scale = None
    for path in [
        os.path.join(ckpt_dir, f"logit_scale_bval.pt"),
        os.path.join(ckpt_dir, "logit_scale.pt")
    ]:
        if os.path.isfile(path):
            logit_scale = path
            break
    
    return vision_ckpt, vpt_tokens, text_feats, logit_scale

def load_model(num_prompt, ckpt_dir, device, pe_vision_config="PE-Core-L14-336"):
    """Load and configure model."""
    vision_ckpt, vpt_tokens, text_path, logit_scale = discover_checkpoints(ckpt_dir)
    
    
    model = PECore_Vision(
        vision_cfg=PE_VISION_CONFIG[pe_vision_config],
        num_prompt=num_prompt
    )
    model.load_baseline(vision_ckpt, device)    
        
    # Load VPT tokens if needed
    if num_prompt > 0 and vpt_tokens:
        num_tokens = 3
        for token_path in vpt_tokens[:num_tokens]:
            try:
                model.load_VPT_token(token_path, device)
            except Exception as e:
                print(f"Warning: failed to load VPT token: {e}")

    if logit_scale:
        try:
            loaded_scale = torch.load(logit_scale, map_location=device)
            model.logit_scale = torch.nn.Parameter(loaded_scale)
            print(f"Loaded logit scale from checkpoint {loaded_scale}")
        except Exception as e:
            print(f"Warning: failed to load logit scale: {e}")
    
    # Load the text features if available
    if not text_path or not os.path.isfile(text_path):
        return None
    try:
        obj = torch.load(text_path, map_location=device)
        if isinstance(obj, dict) and "text_features" in obj:
            # If saved as state_dict than load the tensor
            text_features = obj["text_features"]
        elif torch.is_tensor(obj):
            text_features = obj
    except Exception as e:
        print(f"Warning: failed to load text features: {e}")
    
    if hasattr(model, "text_features") and text_features is not None:
        model.text_features = text_features
    else:
        raise ValueError("The model does not have a text_features attribute or text_features is None.")
    
    return model


def compute_concordance_index(baseline_preds, model_preds):
    ''' 
    Compute the concordance index between two sets of predictions.    
    '''
    if len(baseline_preds) != len(model_preds):
        raise ValueError("The two prediction lists must have the same length.")
    
    concordance = 0
    discordance = 0

    for i in range(len(baseline_preds)):
        if baseline_preds[i] == model_preds[i]:
            concordance += 1
        else:
            discordance += 1

    if concordance + discordance == 0:
        return 0.0
    c_index = concordance / (concordance + discordance)
    return {
        'c_index': c_index,
        'discordance': discordance,
        'concordance': concordance,
        'total': concordance + discordance,
    }

    
def get_baseline_text_features(model, tokenizer, text_list, device):

    '''
    Compute the text features from a given text using the provided model and tokenizer.
    '''
    text_features = []

    for text in text_list:
        if type(text) == list:
            for t in text:
                text_features.append(tokenizer(t))
        else:
            text_features.append(tokenizer(text))

    text_features = torch.cat(text_features).to(device)
    print("Tokenized text shape:", text_features.shape)
    with torch.no_grad():
        text_features = model.encode_text(text_features, normalize=True)
    print("Text features shape:", text_features.shape)

    return text_features



def compute_baseline_prediction(dataset, args):
    print("Loading baseline model...")
    baseline = CLIP.from_config("PE-Core-L14-336", pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline.to(device)
    baseline.eval()

    print("Encoding text prompts...")
    text_features = get_baseline_text_features(baseline, PETokenizer(32), TEXT_PROMPTS, device)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    iterator = tqdm(dataloader, desc="Baseline Inference", unit="batch") if args.tqdm else dataloader
    age, gender, emotion = [], [], []
    print("Starting inference...")
    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device)
            image_features = baseline.encode_image(images, normalize=True)
            logits = baseline.logit_scale.exp() * (image_features @ text_features.t())
            for task in torch.split(logits, [9, 2, 7], dim=-1):
                pred = task.argmax(dim=-1).cpu().numpy()
                if task.shape[-1] == 9:
                    age.extend(pred)
                elif task.shape[-1] == 2:
                    gender.extend(pred)
                elif task.shape[-1] == 7:
                    emotion.extend(pred)
    print("Inference completed.")
    print("Final prediction counts: ", len(age), len(gender), len(emotion))
    return age, gender, emotion

def compute_model_prediction(dataset, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.num_prompt, args.ckpt_dir, device, "PE-Core-L14-336")
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    iterator = tqdm(dataloader, desc="Model Inference", unit="batch") if args.tqdm else dataloader
    age, gender, emotion = [], [], []

    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device)
            logits = model(images)
            for task in logits:
                pred = task.argmax(dim=-1).cpu().numpy()
                if task.shape[-1] == 9:
                    age.extend(pred)
                elif task.shape[-1] == 2:
                    gender.extend(pred)
                elif task.shape[-1] == 7:
                    emotion.extend(pred)

    return age, gender, emotion


def save_prediction(file_path, age, gender, emotion):
    with open(file_path, 'w') as f:
        for i in range(len(age)):
            f.write(f"{age[i]},{gender[i]},{emotion[i]}\n")

def load_prediction(file_path):
    age, gender, emotion = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            a, g, e = line.strip().split(',')
            age.append(int(a))
            gender.append(int(g))
            emotion.append(int(e))

    return age, gender, emotion

def main(args):    
    print("Loading dataset...")
    dataset = BaseDataset(
        args.dataset_path,
        get_image_transform(336),
        split="test",
        verbose=False
    )


    
    if args.dataset_path.endswith('/') or args.dataset_path.endswith('\\'):
        args.dataset_path = args.dataset_path[:-1]

    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    model_considered = os.path.basename(os.path.normpath(args.ckpt_dir.split("ckpt")[0]))

    print(f'Dataset loaded: {dataset_name} with {len(dataset)} samples.')
    baseline_prediction = None
    model_prediction = None

    output_dir = os.path.join(args.output_dir, dataset_name)
    prediction_dir = os.path.join(args.output_dir, "predictions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    #check if the prediction are already available for the models
    os.makedirs(prediction_dir, exist_ok=True)
    if os.path.exists(os.path.join(prediction_dir, f"baseline_{dataset_name}.txt")):
        print(f"Found existing baseline predictions for {dataset_name} dataset, loading...")
        baseline_prediction = load_prediction(os.path.join(prediction_dir, f"baseline_{dataset_name}.txt"))
    else:
        print(f"No existing baseline predictions for {dataset_name} dataset, computing from scratch...")
        baseline_prediction = compute_baseline_prediction(dataset, args)
        save_prediction(os.path.join(prediction_dir, f"baseline_{dataset_name}.txt"), *baseline_prediction)
    if os.path.exists(os.path.join(prediction_dir, f"{model_considered}_{dataset_name}.txt")):
        print(f"Found existing predictions for model {model_considered} on {dataset_name} dataset, loading...")
        model_prediction = load_prediction(os.path.join(prediction_dir, f"{model_considered}_{dataset_name}.txt"))
    else:
        print(f"No existing predictions for model {model_considered} on {dataset_name} dataset, computing from scratch...")
        model_prediction = compute_model_prediction(dataset, args)
        save_prediction(os.path.join(prediction_dir, f"{model_considered}_{dataset_name}.txt"), *model_prediction)

    c_index = []
    concordance = []
    discordance = []
    total = []
    for i in range(len(baseline_prediction)):
        result = compute_concordance_index(
            baseline_prediction[i], model_prediction[i]
        )
        c_index.append(result['c_index'])
        concordance.append(result['concordance'])
        discordance.append(result['discordance'])
        total.append(result['total'])


    headers = ["Task", "Concordance Index"]
    rows = [
        ["Age", c_index[0]],
        ["Gender", c_index[1]],
        ["Emotion", c_index[2]],
    ]
    
    print(f"Concordance Index of model {model_considered} on dataset {dataset_name}:")
    print(format_table(headers, rows))
    
    with open(os.path.join(output_dir, f"{model_considered}_c_index.txt"), 'w') as f:
        f.write("This file contains the concordance index results between the baseline and a multitask model.")
        f.write("The concordance index is computed as the ratio of the number of agreements over the total number of comparisons made.\n\n")
        f.write("#"*10 + f" Concordance Index of model {model_considered} on dataset {dataset_name}:\n")
        f.write(f"- Concordance count:\n\t--Age: {concordance[0]} out of {total[0]}\n")
        f.write(f"\t--Gender: {concordance[1]} out of {total[1]}\n")
        f.write(f"\t--Emotion: {concordance[2]} out of {total[2]}\n")
        f.write(f"- Discordance count:\n\t--Age: {discordance[0]} out of {total[0]}\n")
        f.write(f"\t--Gender: {discordance[1]} out of {total[1]}\n")
        f.write(f"\t--Emotion: {discordance[2]} out of {total[2]}\n")
        f.write("\n")            
        f.write(format_table(headers, rows) + "\n\n")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute Concordance Index between Baseline and Model Predictions")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--output_dir", type=str, default="./concordance_results", help="Directory to save the concordance index results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm progress bars.")
    parser.add_argument("--num_prompt", type=int, default=0, help="Number of prompt tokens to use in the model.")
    args = parser.parse_args()
    
    main(args)