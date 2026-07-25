from datasets import load_dataset
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from transformers import Trainer , TrainingArguments
import numpy as np

dataset = load_dataset("fancyzhx/ag_news")
print(dataset['train'].features)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128 )

tokenized_dataset = dataset.map(tokenize , batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label","labels")
tokenized_dataset = tokenized_dataset.remove_columns(["text"])


train_data = tokenized_dataset['train'].select(range(5000))
test_data = tokenized_dataset['test'].select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels= 4)

def compute_metrics(eval_pred):
    prediction , labels = eval_pred
    prediction = np.argmax(prediction , axis = 1)
    accuracy = (prediction==labels).mean()

    return {"Accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="/results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_data,
    eval_dataset= test_data,
    compute_metrics=compute_metrics
)   
trainer.train()

result = trainer.evaluate()
print(result)