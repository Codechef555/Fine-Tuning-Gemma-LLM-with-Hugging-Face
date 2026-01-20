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

* Checks CUDA availability and GPU memory
* Ensures sufficient resources for model fine-tuning

### 2. Dataset Preparation

* Handles raw text related to food content
* Maps short tags to meaningful labels, such as:

  * Nutrition panels
  * Ingredient lists
  * Recipes
  * Menus
  * Food & drink items

### 3. Prompt Engineering

* Converts raw examples into **LLM-compatible chat formats**
* Uses system and user roles to guide the model
* Ensures consistent and structured outputs

### 4. Structured Output Generation

* Extracts information into clearly defined fields
* Example outputs include:

  * `food_items`: list of detected food items
  * `drink_items`: list of detected beverages

### 5. Baseline Inference

* Runs inference using the **base Gemma model**
* Compares default outputs with expected structured responses

### 6. Fine-Tuning Pipeline (In Progress)

* Integration with **TRL** for supervised fine-tuning
* Prepared for instruction-based training
* Scalable for future reinforcement learning or preference tuning

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
* 🚧 Fine-tuning loop
* 🚧 Evaluation metrics
* 🚧 Model deployment

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
