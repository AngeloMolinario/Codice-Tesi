# Prompt-Optimized Vision-Language Models for Facial Attribute Recognition

[cite_start]This repository contains the official implementation for the experiments described in the thesis, "Prompt optimization technique for VLM in a multitask classification problem"[cite: 8]. [cite_start]The project focuses on adapting modern Vision-Language Models (VLMs) for the efficient multitask classification of **gender**, **age**, and **emotion** from facial images[cite: 11].

## Overview

[cite_start]The core idea is to leverage parameter-efficient prompt optimization techniques to specialize large, pre-trained VLMs (like SigLip2 and Perception Encoder 'core') for facial analysis without altering their internal weights[cite: 13, 45]. [cite_start]This avoids the high costs and potential "catastrophic forgetting" associated with full fine-tuning[cite: 36, 47]. [cite_start]We explore both textual and visual prompting strategies to guide the model's focus and enhance its performance on these specific tasks[cite: 12, 48].

***

## Methodology

[cite_start]The project centers on comparing different prompt tuning methods[cite: 37]. [cite_start]These techniques involve optimizing a small set of "soft" prompt parameters that are added to the model's input, rather than retraining the entire model[cite: 38, 39].

### Textual Prompt Tuning

[cite_start]We will implement **Soft Context Prompt Tuning (SoftCPT)**, focusing on the text encoder of the VLM[cite: 48, 49].
* [cite_start]**Class-Agnostic Task-Specific (CATS)**: In this approach, a small meta-network learns to generate continuous prompt vectors based on a description of the task (e.g., "Identify the expressed emotion")[cite: 50, 51]. [cite_start]This single context is then applied uniformly to all classes within that task[cite: 55].

### Visual Prompt Tuning

[cite_start]We will implement **Vision Prompt Tuning (VPT)**, which introduces learnable prompt vectors directly to the visual encoder[cite: 41, 56].
* [cite_start]**VPT Task-Agnostic**: A single, shared set of visual prompts is optimized for all tasks (gender, age, and emotion) simultaneously[cite: 58]. [cite_start]This is efficient and leverages correlations between tasks, requiring only one forward pass for inference[cite: 59, 81].
* [cite_start]**VPT Task-Specific (VPT-TS)**: Each task gets its own dedicated set of visual prompts[cite: 60, 84]. [cite_start]This allows for greater specialization and potentially higher performance on individual tasks but increases the number of parameters and requires a separate forward pass for each task during inference[cite: 61, 62, 87].

***

## Experimental Setup

[cite_start]To determine the most effective strategy, the following three distinct experiments will be conducted and compared against the baseline **zero-shot performance** of the original VLMs[cite: 75, 76].

1.  [cite_start]**SoftCPT-CATS**: Evaluates the class-agnostic, task-specific textual prompting approach[cite: 77].
2.  [cite_start]**VPT Task-Agnostic**: Evaluates the performance of a single, unified set of visual prompts shared across all tasks[cite: 80].
3.  [cite_start]**VPT-TS**: Evaluates the use of specialized visual prompts for each classification task[cite: 82, 84].

[cite_start]An advanced method, **Visual Variational Autoencoder Prompt Tuning (V2PT)**, may also be explored if time permits[cite: 63, 90]. [cite_start]This technique dynamically generates part of the visual prompt based on the input image itself[cite: 64, 65, 92].

### Evaluation Metrics

[cite_start]Model performance will be assessed based on three key metrics[cite: 67]:
* [cite_start]**Task Accuracy**: The proportion of correct predictions for each classification task[cite: 68, 69].
* [cite_start]**Added Parameters**: The number of extra learnable parameters introduced by the prompt tuning method[cite: 70].
* [cite_start]**Latency**: The inference time required to process an input, measuring the computational overhead of each method[cite: 71, 72].

***

## Datasets

[cite_start]The experiments will utilize a combination of the following public datasets for training and evaluation on gender, age, and emotion recognition[cite: 95]:
* [cite_start]**Celeba-HQ**: 30,000 high-quality celebrity faces labeled with gender[cite: 96].
* [cite_start]**LFW (Labeled Faces in the Wild)**: Contains 13,233 face images labeled for gender[cite: 97, 98].
* [cite_start]**Raf-DB**: Contains 29,672 images tagged with facial expressions[cite: 99, 100].
* [cite_start]**VGGFace2 Test**: Contains 70,909 samples with annotations for age and gender[cite: 102].
* [cite_start]**UTKFace**: Consists of over 24,000 images labeled with age and gender[cite: 103].
* [cite_start]**FairFace Test**: Contains over 10,000 images balanced for race, with gender and age labels[cite: 104, 105].

***

## Demonstrator

[cite_start]The final goal is to integrate the best-performing model into a real-time face detection system, creating a demonstrator capable of analyzing faces from a live image feed and classifying their gender, age, and emotion[cite: 107, 155].
