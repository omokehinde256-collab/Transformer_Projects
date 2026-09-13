from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ==============================
# MODEL CONFIGURATION
# ==============================

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

print("Loading Q&A model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

print("Q&A model loaded successfully!")


# ==============================
# FASTAPI APPLICATION
# ==============================

app = FastAPI(
    title="Transformer Q&A API",
    description="Question Answering API powered by a pretrained DistilBERT model",
    version="1.0.0"
)


# ==============================
# REQUEST FORMAT
# ==============================

class QuestionRequest(BaseModel):
    question: str
    context: str


# ==============================
# HOME ENDPOINT
# ==============================

@app.get("/")
def home():
    return {
        "message": "Transformer Q&A API is running",
        "model": MODEL_NAME
    }


# ==============================
# QUESTION ANSWERING
# ==============================

@app.post("/answer")
def answer_question(request: QuestionRequest):

    inputs = tokenizer(
        request.question,
        request.context,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_position = torch.argmax(
        outputs.start_logits,
        dim=1
    ).item()

    end_position = torch.argmax(
        outputs.end_logits,
        dim=1
    ).item()

    # Prevent invalid span
    if end_position < start_position:
        end_position = start_position

    answer_tokens = inputs["input_ids"][0][
        start_position:end_position + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    start_score = torch.softmax(
        outputs.start_logits,
        dim=1
    )[0][start_position].item()

    end_score = torch.softmax(
        outputs.end_logits,
        dim=1
    )[0][end_position].item()

    confidence = (start_score + end_score) / 2

    return {
        "question": request.question,
        "answer": answer,
        "confidence": round(confidence, 4),
        "start_position": start_position,
        "end_position": end_position
    }