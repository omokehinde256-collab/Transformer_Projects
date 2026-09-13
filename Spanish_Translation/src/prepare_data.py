from datasets import load_dataset

print("Loading Spanish-English dataset...")

dataset = load_dataset(
    "Helsinki-NLP/opus_books",
    "en-es"
)

print("\nDataset loaded successfully!")
print(dataset)

print("\nFirst example:")
print(dataset["train"][0])

# Shuffle the dataset
dataset = dataset["train"].shuffle(seed=42)

# Use 20,000 examples for our first CPU fine-tuning run
dataset = dataset.select(range(min(20000, len(dataset))))

# Split into training and validation
split = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split["train"]
validation_dataset = split["test"]

print("\nTraining examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))

# Save locally
train_dataset.save_to_disk("data/train")
validation_dataset.save_to_disk("data/validation")

print("\nDataset preparation complete!")