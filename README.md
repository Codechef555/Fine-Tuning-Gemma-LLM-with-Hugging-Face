# Fine-Tuning Gemma LLM with Hugging Face

## Description

This project focuses on **fine-tuning the Gemma Large Language Model (LLM)** using the **Hugging Face ecosystem**. The goal is to adapt a pre-trained Gemma model to a **domain-specific text classification and information extraction task**, particularly around **food-related content** such as recipes, menus, ingredients, nutrition panels, and advertisements.

The project is currently **~50% complete** and includes data preprocessing, prompt engineering, dataset formatting, and baseline inference using the base Gemma model. The fine-tuning pipeline is being built using **Transformers**, **TRL (Transformers Reinforcement Learning)**, **Datasets**, and **Accelerate**.

---

## Overview

Large Language Models perform well out-of-the-box, but domain-specific accuracy can be significantly improved through fine-tuning. In this project:

* The **Gemma model** is used as the base LLM
* Custom datasets are transformed into **chat-style prompts** compatible with instruction-tuned models
* Outputs are structured to extract **food and drink-related entities**
* The project is designed to be scalable for future tasks such as tagging, filtering, and classification

This repository serves as both a **learning reference** and a **production-ready foundation** for fine-tuning Gemma using Hugging Face tools.

---

## Architecture

```
┌────────────────────┐
│  Raw Text Dataset  │
│ (Food-related data)│
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Data Preprocessing │
│ - Cleaning         │
│ - Label Mapping    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────┐
│ Prompt Engineering     │
│ - Chat-style templates │
│ - System/User roles    │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Hugging Face Datasets  │
│ - Train/Test split     │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Gemma Base Model       │
│ (Transformers)         │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Fine-Tuning Pipeline   │
│ - TRL                  │
│ - Accelerate           │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Inference & Evaluation │
└────────────────────────┘
```

---

## Functionalities

### 1. GPU & Environment Validation

* Verifies CUDA availability and GPU memory
* Ensures compatibility for Gemma fine-tuning

### 2. Dataset Preparation

* Processes food-related text datasets
* Converts raw labels into human-readable classes
* Splits data into training and evaluation sets

### 3. Prompt Engineering

* Builds instruction-style chat prompts
* Uses system/user roles compatible with Gemma
* Enforces structured outputs for consistency

### 4. Supervised Fine-Tuning (SFT)

* Implements **SFTTrainer** from TRL
* Tokenizes prompts and responses correctly
* Supports configurable training arguments

### 5. Model Saving & Hub Integration

* Saves fine-tuned checkpoints locally
* Pushes trained model and tokenizer to Hugging Face Hub

### 6. Inference & Validation

* Runs post-training inference on sample inputs
* Compares fine-tuned outputs against baseline responses

---

## Tech Stack

* **Model**: Gemma (Google)
* **Frameworks**:

  * Hugging Face Transformers
  * Hugging Face Datasets
  * TRL
  * Accelerate
* **Language**: Python
* **Interface (Planned)**: Gradio

---

## Project Status

* ✅ Environment setup
* ✅ Dataset formatting
* ✅ Prompt engineering
* ✅ Baseline inference
* ✅ Supervised fine-tuning using TRL (SFTTrainer)
* ✅ Model saving and Hugging Face Hub integration
* 🚧 Evaluation metrics and benchmarking
* 🚧 Demo & deployment

---

## Future Enhancements

* Complete supervised fine-tuning
* Add evaluation metrics (precision, recall, F1)
* Save and publish fine-tuned model to Hugging Face Hub
* Build a Gradio-based demo interface
* Extend tagging to additional domains

---

## Disclaimer

This project is for **educational and experimental purposes** and is still under active development.

---

## Author

**Name : Md.Karaamathullah sheriff**
**email : mdkaraamathullahsheriff@gmail.com**

Feel free to contribute, open issues, or suggest improvements.
