from core.vision_encoder.config import PE_VISION_CONFIG
from core.vision_encoder import transforms
from wrappers.PerceptionEncoder.pe import PECore_Vision
from PIL import Image
import torch
import os

CLASSES =[
        ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        ["male", "female"],
        ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    ]

def get_prediction(age, gender, emotion):
    age_logit = age
    gender_logit = gender
    emotion_logit = emotion


    age_prob = torch.softmax(age_logit, dim=1)
    age_pred = torch.round(torch.sum(age_prob*torch.arange(0, 9, device=age_prob.device).unsqueeze(0), dim=1))
    gender_pred = torch.argmax(gender_logit, dim=1)
    emotion_pred = torch.argmax(emotion_logit, dim=1)
    return int(age_pred.item()), int(gender_pred.item()), int(emotion_pred.item())


output_dir = "../TRAIN/SoftCPT16"
img = "/user/amolinario/processed_datasets/datasets_with_standard_labels/RAF-DB/test/images/00011520.jpg"
# 1. Configurazione e percorsi file
vision_cfg = PE_VISION_CONFIG["PE-Core-B16-224"]
num_prompt = 10
num_classes = (9, 2, 7)
device = "cuda"

vision_ckpt = os.path.join(output_dir, "ckpt", "pecore_vision.pth")  # Modifica il percorso se necessario
text_features_path = os.path.join(output_dir, "ckpt", "temp_text_features.pt")
vpt_files = [
    os.path.join(output_dir, "ckpt", "temp_.pt"),
    os.path.join(output_dir, "ckpt", "temp_1.pt"),
    os.path.join(output_dir, "ckpt", "temp_.pt")
]

# 2. Istanzia il modello con 3 VPT
model = PECore_Vision(
    vision_cfg=vision_cfg,
    num_prompt=num_prompt,
    num_classes=num_classes,
    num_vpt=len(vpt_files)
).to(device)

# 3. Carica vision encoder pretrained (pesi di base, senza VPT)
model.load_pretrained_vision_encoder(name="PE-Core-B16-224")#, checkpoint_path=vision_ckpt)

# 4. Carica i 3 VPT da tre file diversi
model.load_vpt_checkpoints(vpt_files)

# 5. Carica le text features precompute
model.load_text_features(text_features_path, device=device)

# 6. Inferenza: ottieni una lista di output, uno per ogni contesto VPT
model.eval()
image = Image.open(img).convert("RGB")
image = transforms.get_image_transform(224)(image).unsqueeze(0).to(device)  # Prepara l'immagine
with torch.no_grad():
    logits_list = model(image)
print(f"Logits list length: {len(logits_list)}")  # Dovrebbe essere 3, uno per ogni VPT

# logits_list[0]: output con vpt_age, logits_list[1]: output con vpt_gender, ecc.

i = 0
for item in logits_list:
    print(f"VPT {i} - Logits shape: {item.shape}")
    i += 1

pred1 = get_prediction(logits_list[0], logits_list[1], logits_list[2])
print(f"Predictions for VPT {i}: Age: {CLASSES[0][pred1[0]]}, Gender: {CLASSES[1][pred1[1]]}, Emotion: {CLASSES[2][pred1[2]]}")


vision_ckpt = os.path.join(output_dir, "ckpt", "pecore_vision.pth")  # Modifica il percorso se necessario
text_features_path = os.path.join(output_dir, "ckpt", "temp_text_features.pt")
vpt_files = [
    os.path.join(output_dir, "ckpt", "temp_.pt")]

# 2. Istanzia il modello con 3 VPT
model = PECore_Vision(
    vision_cfg=vision_cfg,
    num_prompt=num_prompt,
    num_classes=num_classes,
    num_vpt=len(vpt_files)
).to(device)

# 3. Carica vision encoder pretrained (pesi di base, senza VPT)
model.load_pretrained_vision_encoder(name="PE-Core-B16-224")#, checkpoint_path=vision_ckpt)

# 4. Carica i 3 VPT da tre file diversi
model.load_vpt_checkpoints(vpt_files)

# 5. Carica le text features precompute
model.load_text_features(text_features_path, device=device)

# 6. Inferenza: ottieni una lista di output, uno per ogni contesto VPT
model.eval()
image = Image.open(img).convert("RGB")
image = transforms.get_image_transform(224)(image).unsqueeze(0).to(device)  # Prepara l'immagine
with torch.no_grad():
    logits_list = model(image)
print(f"Logits list length: {len(logits_list)}")  # Dovrebbe essere 3, uno per ogni VPT

# logits_list[0]: output con vpt_age, logits_list[1]: output con vpt_gender, ecc.

i = 0
for item in logits_list:
    print(f"VPT {i} - Logits shape: {item.shape}")
    i += 1

pred2 = get_prediction(logits_list[0], logits_list[1], logits_list[2])



vision_ckpt = os.path.join(output_dir, "ckpt", "pecore_vision.pth")  # Modifica il percorso se necessario
text_features_path = os.path.join(output_dir, "ckpt", "temp_text_features.pt")
vpt_files = []

# 2. Istanzia il modello con 3 VPT
model = PECore_Vision(
    vision_cfg=vision_cfg,
    num_prompt=0,
    num_classes=num_classes,
    num_vpt=len(vpt_files)
).to(device)

# 3. Carica vision encoder pretrained (pesi di base, senza VPT)
model.load_pretrained_vision_encoder(name="PE-Core-B16-224")#, checkpoint_path=vision_ckpt)

# 4. Carica i 3 VPT da tre file diversi
model.load_vpt_checkpoints(vpt_files)

# 5. Carica le text features precompute
model.load_text_features(text_features_path, device=device)

# 6. Inferenza: ottieni una lista di output, uno per ogni contesto VPT
model.eval()
image = Image.open(img).convert("RGB")
image = transforms.get_image_transform(224)(image).unsqueeze(0).to(device)  # Prepara l'immagine
with torch.no_grad():
    logits_list = model(image)
print(f"Logits list length: {len(logits_list)}")  # Dovrebbe essere 3, uno per ogni VPT

# logits_list[0]: output con vpt_age, logits_list[1]: output con vpt_gender, ecc.

i = 0
for item in logits_list:
    print(f"VPT {i} - Logits shape: {item.shape}")
    i += 1


pred3 = get_prediction(logits_list[0], logits_list[1], logits_list[2])



############################## SOFTCPT ###################################à


vision_ckpt = os.path.join(output_dir, "ckpt", "pecore_vision.pth")  # Modifica il percorso se necessario
text_features_path = os.path.join(output_dir, "ckpt", "temp_text_features.pt")
vpt_files = []

# 2. Istanzia il modello con 3 VPT
model = PECore_Vision(
    vision_cfg=vision_cfg,
    num_prompt=0,
    num_classes=num_classes,
    num_vpt=len(vpt_files)
).to(device)

# 3. Carica vision encoder pretrained (pesi di base, senza VPT)
model.load_pretrained_vision_encoder(name="PE-Core-B16-224")#, checkpoint_path=vision_ckpt)

# 4. Carica i 3 VPT da tre file diversi
model.load_vpt_checkpoints(vpt_files)

# 5. Carica le text features precompute
model.load_text_features(text_features_path, device=device)

# 6. Inferenza: ottieni una lista di output, uno per ogni contesto VPT
model.eval()
image = Image.open(img).convert("RGB")
image = transforms.get_image_transform(224)(image).unsqueeze(0).to(device)  # Prepara l'immagine
with torch.no_grad():
    logits_list = model(image)
print(f"Logits list length: {len(logits_list)}")  # Dovrebbe essere 3, uno per ogni VPT

# logits_list[0]: output con vpt_age, logits_list[1]: output con vpt_gender, ecc.

i = 0
for item in logits_list:
    print(f"VPT {i} - Logits shape: {item.shape}")
    i += 1


pred4 = get_prediction(logits_list[0], logits_list[1], logits_list[2])

print(f"Prediction for soft CPT {i}: Age {CLASSES[0][pred4[0]]}, Gender {CLASSES[1][pred4[1]]}, Emotion {CLASSES[2][pred4[2]]}")
print(f"Prediction for multitask VPT {i}: Age {CLASSES[0][pred2[0]]}, Gender {CLASSES[1][pred2[1]]}, Emotion {CLASSES[2][pred2[2]]}")
print(f"Predictions for 3 VPT {i}: Age: {CLASSES[0][pred1[0]]}, Gender: {CLASSES[1][pred1[1]]}, Emotion: {CLASSES[2][pred1[2]]}")
print(f"Predictions 0 VPT {i}: Age: {CLASSES[0][pred3[0]]}, Gender: {CLASSES[1][pred3[1]]}, Emotion: {CLASSES[2][pred3[2]]}")
