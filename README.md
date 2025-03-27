# Lightweight Fine-Tuning of a Foundation Model using Hugging Face PEFT (LoRA)

This project is part of the **Bertelsmann Next Generation Tech Booster** and **Udacity**. It focuses on applying **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** to fine-tune a Hugging Face transformer model for a sequence classification task.

## ✅ Project Status: Passed Review
> *"You have demonstrated a very good understanding of foundational models and Parameter Efficient Fine Tuning... Keep up all the great work!"*  
> — *Udacity Reviewer Feedback*

---

## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Tech Stack](#tech-stack)
- [Key Concepts](#key-concepts)
- [Steps Implemented](#steps-implemented)
- [Evaluation](#evaluation)
- [Reviewer Feedback](#reviewer-feedback)
- [Future Improvements](#future-improvements)
- [How to Reuse This Project](#how-to-reuse-this-project)

---

## Overview

Fine-tuning large models can be computationally expensive and often unnecessary. **PEFT** offers a lightweight solution by fine-tuning a small number of model parameters—achieving efficiency and performance gains at lower cost. This project uses **LoRA adapters** from the Hugging Face PEFT library to fine-tune a GPT-2 model for sentiment classification.

---

## Objective

- Load a pre-trained Hugging Face model and evaluate it
- Fine-tune the model using **LoRA**
- Evaluate the fine-tuned model and compare performance
- Save and reuse the PEFT model for inference

---

## Tech Stack

- Python 3
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT
- PyTorch
- LoRA (Low-Rank Adaptation)

---

## Key Concepts

- **PEFT (Parameter Efficient Fine-Tuning)**: Fine-tuning a small number of parameters using adapters.
- **LoRA (Low-Rank Adaptation)**: A PEFT technique that injects trainable low-rank matrices into each layer.
- **Trainer API**: Used for training and evaluation.

---

## Steps Implemented

### 1. Load Foundation Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### 2. Load Dataset
```python
from datasets import load_dataset
dataset = load_dataset("imdb").shuffle(seed=42).select(range(1000))  # sample subset
```

### 3. Evaluate Base Model

Used the Hugging Face Trainer to compute accuracy before fine-tuning.
Base model accuracy: ~20%


---

### 4. Apply LoRA Fine-Tuning
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(task_type="SEQ_CLS", inference_mode=False)
peft_model = get_peft_model(model, config)
```
### 5. Train PEFT Model

Model was trained for 1 epoch using the Hugging Face Trainer.
Saved weights to: ./peftmodel-gpt2-sentiment/


---

### 6. Load and Evaluate Fine-Tuned Model
```python
from peft import AutoPeftModelForSequenceClassification
peft_model = AutoPeftModelForSequenceClassification.from_pretrained("peftmodel-gpt2-sentiment")
```

Post-fine-tuning accuracy: ~55%
(Improved from ~20%, though still low in practical terms.)


---

### Reviewer Feedback

✅ Positives

Correct use of PEFT and LoRA

Accurate loading, training, saving, and reloading of models

Excellent organization and effort

Followed reviewer suggestions from previous review

⚠️ Suggestions

Use Better Metrics: Accuracy may be misleading for imbalanced datasets.

Explore Class Balance: Consider distribution of classes before choosing metrics.

Improve Performance:

Try different LoRA configs

Increase training epochs

Try more advanced evaluation metrics (e.g., F1, precision/recall)




---

### Future Improvements

Implement QLoRA (quantized LoRA) for faster and lighter training

Try additional PEFT techniques supported by Hugging Face

Conduct more extensive hyperparameter tuning

Use confusion matrix and F1 score for imbalanced datasets



---

### How to Reuse This Project

# 1. Clone the repo:
```
git clone https://github.com/yourusername/peft-lora-finetuning
cd peft-lora-finetuning
```

# 2. Install dependencies:
```
pip install -r requirements.txt
```

# 3. Run the notebook: Open peft_lora_finetuning.ipynb and step through each cell.


# 4. Load fine-tuned model:
```python
from peft import AutoPeftModelForSequenceClassification
model = AutoPeftModelForSequenceClassification.from_pretrained("peftmodel-gpt2-sentiment")
```

---

### 🎓 Supported By

<a href="https://www.udacity.com/bertelsmann-tech-scholarships" target="_blank">
  <img src="https://user-images.githubusercontent.com/85645859/235321029-5bb6d4d3-0736-4bd1-8264-eaf6e4c3b94e.png" alt="Bertelsmann Next Gen Tech Booster" height="60"/>
</a>

This project was completed as part of the **Bertelsmann Next Generation Tech Booster Scholarship**, powered by **Udacity**.

---
