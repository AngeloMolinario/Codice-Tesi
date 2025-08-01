# 🗂️ wrappers/promptopt

This folder contains the code for **prompt optimization** techniques used in the project [Prompt-Optimized Vision-Language Models for Facial Attribute Recognition](../README.md). The code here is designed to implement, experiment with, and extend prompt-based adaptation methods for Vision-Language Models (VLMs), focusing on facial attribute recognition tasks such as gender, age, and emotion classification.

## 📦 What is in this folder?

- **Prompt Tuning Implementations:**  
  Source code for both textual and visual prompt optimization strategies:
  - **Textual Prompt Tuning (SoftCPT):** Methods to adapt the textual encoder using learnable prompt vectors, as described in the main repository.
  - **Visual Prompt Tuning (VPT):** Components for adapting the visual encoder via learnable image patch prompts, including both task-agnostic and task-specific modes.

- **Experimentation Tools:**  
  Scripts and utilities to run experiments, evaluate models, and compare prompt optimization strategies on various facial analysis benchmarks.

- **Supporting Modules:**  
  Helper functions, configuration files, and model wrappers to streamline integration of prompt optimization into the overall VLM pipeline.

## 🔗 Reference to Base Repository

For a full overview of the project’s goals, methodology, and experimental setup, see the [main README](../../README.md) at the root of this repository.

The prompt optimization techniques implemented here are central to the thesis:
> **"Prompt Optimization Technique for VLM in a Multitask Classification Problem"**

This folder operationalizes the key approaches described, enabling parameter-efficient adaptation of state-of-the-art VLMs for multitask facial attribute recognition.

---

**For details on experiments, datasets, and evaluation metrics, refer to the root [README.md](../../README.md).**
