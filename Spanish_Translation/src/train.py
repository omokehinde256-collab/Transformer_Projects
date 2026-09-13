from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"

print("Loading datasets...")

train_dataset = load_from_disk("data/train")
validation_dataset = load_from_disk("data/validation")

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained Spanish-English model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


def preprocess_function(examples):
    translations = examples["translation"]

    spanish_texts = [
        item["es"]
        for item in translations
    ]

    english_texts = [
        item["en"]
        for item in translations
    ]

    model_inputs = tokenizer(
        spanish_texts,
        max_length=128,
        truncation=True,
    )

    labels = tokenizer(
        text_target=english_texts,
        max_length=128,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing training dataset...")

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("Training dataset tokenized!")

print("\nTokenizing validation dataset...")

tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=validation_dataset.column_names,
)

print("Validation dataset tokenized!")


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


training_args = Seq2SeqTrainingArguments(
    output_dir="models/spanish_to_english",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=2,
    predict_with_generate=True,
    logging_steps=100,
    report_to="none",
    fp16=False,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
)


print("\n==============================")
print("STARTING FINE-TUNING")
print("==============================")

trainer.train()

print("\nFine-tuning completed!")

print("\nSaving fine-tuned model...")

trainer.save_model("models/spanish_to_english")
tokenizer.save_pretrained("models/spanish_to_english")

print("\nModel saved successfully!")
print("Location: models/spanish_to_english")