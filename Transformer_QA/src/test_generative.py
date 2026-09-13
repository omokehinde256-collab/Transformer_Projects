from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Loading FLAN-T5...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")

context = """
The Federal University of Technology, Akure (FUTA) is a public
university located in Akure, Ondo State, Nigeria. The university
was established in 1981 with the aim of developing highly skilled
manpower in science, engineering, and technology.
"""

question = "When was FUTA established?"

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n==============================")
print("GENERATIVE QA RESULT")
print("==============================")
print("Question:", question)
print("Answer:", answer)
print("==============================")
