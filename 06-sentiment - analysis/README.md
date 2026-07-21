# Project 6: NLP with Transformers

## Project Overview

In this project, I built a sentiment analysis model using a pretrained Transformer model called **DistilBERT**. The model was already trained on a large amount of text, and I fine-tuned it on the IMDb movie review dataset to classify reviews as **Positive** or **Negative**.

To reduce the training time and understand the Hugging Face workflow, I used only a small part of the dataset (500 training samples and 100 testing samples).

---

## Dataset

**Source:** `stanfordnlp/imdb`

- Original dataset: 50,000 movie reviews
- Training samples used: 500
- Testing samples used: 100
- Classes: Positive and Negative

---

## Model

- Pretrained model: **DistilBERT**
- Framework: Hugging Face Transformers
- Task: Binary Sentiment Classification
- Fine-tuned for 1 training epoch

---

## Results

- **Accuracy:** 80%
- **Training Epochs:** 1

---

## Key Findings

- Even with a small dataset, the pretrained Transformer achieved good results.
- Hugging Face makes it easy to fine-tune pretrained language models.
- Using only 500 training samples reduced the training time while still giving an accuracy of around **80%**.
- The model can correctly classify most movie reviews as positive or negative.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- Hugging Face documentation

---

## How to Run

```bash
python sentiments.py
```