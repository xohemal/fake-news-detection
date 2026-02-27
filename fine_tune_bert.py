import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import numpy as np
import evaluate

# Load and map labels to 6 classes
def load_data(path='train.tsv'):
    cols = ['id','label','statement','subject','speaker','job_title','state','party',
            'barely_true','false','half_true','mostly_true','pants_on_fire','context']
    df = pd.read_csv(path, sep='\t', names=cols)
    label_map = {
        'pants-fire': 0,
        'false': 1,
        'barely-true': 2,
        'half-true': 3,
        'mostly-true': 4,
        'true': 5
    }
    df = df[df['label'].isin(label_map.keys())]
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['statement'])
    return df[['statement', 'label']]

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Prepare datasets
df = load_data()
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

def tokenize_function(examples):
    return tokenizer(examples["statement"], truncation=True, padding=True)

train_ds = train_ds.map(tokenize_function, batched=True)
valid_ds = valid_ds.map(tokenize_function, batched=True)

# Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Training args
training_args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train and Save
trainer.train()
trainer.save_model("bert_model")
tokenizer.save_pretrained("bert_model")
