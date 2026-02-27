from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1
)

print("✅ TrainingArguments accepted!")
