# 🧠 Prompt-Optimized Vision-Language Models for Facial Attribute Recognition

This repository contains the official implementation of the thesis:  
**"Prompt Optimization Technique for VLM in a Multitask Classification Problem"**

---

## 🚀 Overview

This thesis focuses on adapting a pretrained Vision-Language Model (VLM) to a **multitask classification problem**, aimed at recognizing **gender**, **age**, and **emotion** from facial images.

The main focus is **prompt optimization** — both **textual** and **visual** — to guide the model’s attention to the most relevant features for each task.  
Through prompt tuning, the model can be adapted in a **parameter-efficient** way without modifying its internal weights, leveraging the knowledge gained during pre-training.

---

## 🛠️ Methodology

The project develops a multitask model for gender, age, and emotion classification using state-of-the-art VLMs such as:

- **SigLIP2**
- **Perception Encoder Core (PE-Core)**

### Key Approach:
**Prompt optimization techniques** to specialize pretrained models, avoiding full fine-tuning which is computationally expensive.

We explore two complementary strategies:

---

### ✍️ Textual Prompt Tuning (SoftCPT)

SoftCPT focuses on **tuning the textual encoder**.

- A **shared meta-network** (text encoder + MLP) generates **task-specific continuous prompts**.
- The task description (e.g., _"Classify the person’s gender"_) is encoded.
- Learnable continuous vectors are concatenated with this embedding.
- The final prompt is used with class embeddings during classification.

**Mode:**  
- **Class-Agnostic Task-Specific (CATS)**: context vectors are generated from task descriptions and shared across all classes of that task.

---

### 🖼️ Visual Prompt Tuning (VPT)

VPT focuses on **tuning the visual encoder**. We explore two modes:

#### VPT Task-Agnostic:
- One set of learnable visual prompts shared across **all tasks** and **all classes**.
- Efficient when tasks are strongly correlated.
- Single forward pass per input → faster inference.

#### VPT Task-Specific (VPT-TS):
- Different visual prompts **per task**, shared across classes.
- Better performance for weakly correlated tasks.
- Requires multiple forward passes (one per task) → more computationally expensive.

---

### 🧪 Experimental Setup

All evaluations are performed on **unseen test data**.  
Each experiment is repeated across the selected models, and compared to a **zero-shot baseline**.

#### Experiments:
- **SoftCPT-CATS**: Class-Agnostic Task-Specific textual prompt tuning.
- **VPT Task-Agnostic**: Shared visual prompts across all tasks.
- **VPT-TS**: Task-specific visual prompts (different per task).

#### Optional:
- **V2PT (Visual Variational Autoencoder Prompt Tuning)**  
  - Prompt vectors = [learned prompts] + [VAE-generated dynamic prompts].

---

## 📈 Evaluation Metrics

- **Task Accuracy**: Correct predictions / Total samples.
- **Added Parameters**: Extra parameters introduced by prompt tuning.
- **Latency**: Inference time per image or batch.

---

## 📚 Datasets

| Dataset         | Description |
|----------------|-------------|
| **CelebA-HQ**   | 30,000 celebrity face images with gender labels. |
| **LFW**         | 13,233 face images from 5,749 people; gender labels. |
| **RAF-DB**      | 29,672 facial images with expression labels. |
| **VGGFace2 Test** | 70,909 images from 207 individuals, labeled for age and gender. |
| **UTKFace**     | 24,102 real-world facial images labeled with age and gender. |
| **FairFace Test** | 10,954 balanced images across ethnicities with age and gender labels. |

---

## 💡 Demonstrator

We aim to integrate the trained model into a **face detection pipeline**, enhancing it with:

- **Gender classification**
- **Age estimation**
- **Emotion recognition**

---
