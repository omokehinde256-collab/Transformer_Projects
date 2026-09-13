import tkinter as tk
from tkinter import messagebox
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

print("Loading Q&A model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


def answer_question():
    context = context_box.get("1.0", tk.END).strip()
    question = question_box.get().strip()

    if not context:
        messagebox.showwarning("Missing Context", "Please enter some context.")
        return

    if not question:
        messagebox.showwarning("Missing Question", "Please enter a question.")
        return

    inputs = tokenizer(
        question,
        context,
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

    if end_position < start_position:
        end_position = start_position

    answer_tokens = inputs["input_ids"][0][
        start_position:end_position + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    )

    start_probability = torch.softmax(
        outputs.start_logits,
        dim=1
    )[0][start_position].item()

    end_probability = torch.softmax(
        outputs.end_logits,
        dim=1
    )[0][end_position].item()

    confidence = ((start_probability + end_probability) / 2) * 100

    answer_box.config(state="normal")
    answer_box.delete("1.0", tk.END)
    answer_box.insert(
        tk.END,
        f"{answer}\n\nConfidence: {confidence:.2f}%"
    )
    answer_box.config(state="disabled")


# ==========================
# GUI
# ==========================

root = tk.Tk()

root.title("Transformer Q&A System")
root.geometry("800x650")
root.resizable(False, False)

title = tk.Label(
    root,
    text="TRANSFORMER Q&A SYSTEM",
    font=("Arial", 22, "bold")
)

title.pack(pady=20)

subtitle = tk.Label(
    root,
    text="Powered by Hugging Face DistilBERT",
    font=("Arial", 11)
)

subtitle.pack(pady=5)


# Context

context_label = tk.Label(
    root,
    text="Context",
    font=("Arial", 13, "bold")
)

context_label.pack(anchor="w", padx=30, pady=(20, 5))

context_box = tk.Text(
    root,
    height=8,
    width=85,
    font=("Arial", 11),
    wrap="word"
)

context_box.pack(padx=30)


# Question

question_label = tk.Label(
    root,
    text="Question",
    font=("Arial", 13, "bold")
)

question_label.pack(anchor="w", padx=30, pady=(20, 5))

question_box = tk.Entry(
    root,
    width=82,
    font=("Arial", 11)
)

question_box.pack(padx=30, ipady=8)


# Button

ask_button = tk.Button(
    root,
    text="ASK QUESTION",
    font=("Arial", 12, "bold"),
    command=answer_question,
    padx=20,
    pady=10
)

ask_button.pack(pady=25)


# Answer

answer_label = tk.Label(
    root,
    text="Answer",
    font=("Arial", 13, "bold")
)

answer_label.pack(anchor="w", padx=30, pady=(5, 5))

answer_box = tk.Text(
    root,
    height=6,
    width=85,
    font=("Arial", 12),
    wrap="word"
)

answer_box.pack(padx=30)

answer_box.config(state="disabled")


root.mainloop()