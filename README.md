This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **"Prompt Optimization Technique for VLM in a Multitask Classification Problem"** - a thesis focused on adapting pretrained Vision-Language Models (VLMs) for facial attribute recognition tasks (gender, age, emotion).

The project implements prompt optimization techniques for two VLM architectures:
- **SigLIP2**
- **Perception Encoder Core (PE-Core)**

## Core Architecture

### Main Training Scripts
- `coop_train.py` - Single-task prompt tuning with CoOp/VPT methods
- `train_multitask.py` - Multi-task training with shared prompts across tasks
- `test_new.py` - Comprehensive testing and evaluation script

### Key Directories
- `core/` - Core VLM implementations and vision encoders
- `wrappers/` - Model wrappers for SigLIP2, PE-Core, and prompt optimization
- `training/` - Loss functions, training utilities, and optimization logic
- `utils/` - Configuration management, metrics tracking, and utilities
- `dataset/` - Dataset classes and data loading functionality
- `config/` - JSON configuration files for different experimental setups
- `script/` - Shell scripts for running experiments and tests

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```
## Dataset Configuration

Datasets are configured via the `DATASET_NAMES` field in config files:
- Task -1: Multi-task (FairFace, RAF-DB, CelebA_HQ, Lagenda)
- Task 0: Age (FairFace, Lagenda)
- Task 1: Gender (FairFace, RAF-DB, CelebA_HQ, Lagenda)
- Task 2: Emotion (RAF-DB)

Default dataset root: `../processed_datasets/datasets_with_standard_labels`

## Task Classification

The system supports 3 classification tasks:
- **Age**: 9 age groups (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
- **Gender**: Binary (male, female)
- **Emotion**: 7 emotions (surprise, fear, disgust, happy, sad, angry, neutral)

## Prompt Tuning Methods

### Visual Prompt Tuning (VPT)
- `NUM_VISUAL_PROMPT` parameter controls number of visual prompts

### Textual Prompt Tuning (SoftCPT)
- `NUM_TEXT_CNTX` parameter controls text context length

### Context Optimization
- `NUM_TEXT_CNTX` paraemter controls text context length in single task scenario

## Model Wrappers

- `wrappers/SigLip2/` - SigLIP2 model implementation
- `wrappers/PerceptionEncoder/` - PE-Core model wrapper
- `wrappers/promptopt/` - Prompt optimization utilities

## Loss Functions

Located in `training/loss.py`:
- `OrdinalAgeLossEMD` - Earth Mover's Distance loss for age estimation
- `CrossEntropyLoss` - Standard classification loss for gender/emotion
- `MaskedLoss` - Handles missing labels in multi-task scenarios
