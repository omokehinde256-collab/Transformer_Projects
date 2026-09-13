import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained Q&A model...")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

context = """
The University of Notre Dame was founded in 1842 by Rev. Edward Sorin.
It is a private Catholic research university located in Notre Dame, Indiana.
"""

question = "When was the University of Notre Dame founded?"

inputs = tokenizer(
    question,
    context,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

start_position = torch.argmax(start_logits, dim=1).item()
end_position = torch.argmax(end_logits, dim=1).item()

answer_tokens = inputs["input_ids"][0][start_position:end_position + 1]

answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)

print("\n==============================")
print("QUESTION ANSWERING RESULT")
print("==============================")
print("Question:", question)
print("Answer:", answer)
print("Start position:", start_position)
print("End position:", end_position)
print("==============================")