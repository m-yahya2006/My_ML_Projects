from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
import torch



dataset = load_dataset("stanfordnlp/imdb")
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples['text'], truncation =True, padding = True , max_length = 128)

tokenized_dataset = dataset.map(tokenize, batched = True)

small_train = tokenized_dataset['train'].shuffle(seed = 42).select(range(500))
small_test = tokenized_dataset['test'].shuffle(seed = 42).select(range(100))

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2 )


accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    
    predictions = np.argmax(logits, axis=-1)

    return accuracy_metric.compute(predictions=predictions, references=labels)



training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset= small_test,
    compute_metrics = compute_metrics
)

trainer.train()

output =  trainer.evaluate()
print(output)