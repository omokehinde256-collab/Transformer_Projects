import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
import torch

from retriever import Retriever


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EXTRACTIVE_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
GENERATIVE_MODEL = "google/flan-t5-small"


# ============================================================
# MODEL LOADING
# ============================================================

print("Loading Extractive BERT...")

qa_tokenizer = AutoTokenizer.from_pretrained(
    EXTRACTIVE_MODEL
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    EXTRACTIVE_MODEL
)

print("Extractive BERT loaded!")


print("Loading FLAN-T5...")

gen_tokenizer = AutoTokenizer.from_pretrained(
    GENERATIVE_MODEL
)

gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATIVE_MODEL
)

print("FLAN-T5 loaded!")


print("Loading retrieval system...")

retriever = Retriever()

print("Retrieval system loaded!")


# ============================================================
# APPLICATION
# ============================================================

class HybridQAApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Hybrid Transformer Question Answering System"
        )

        self.root.geometry("1000x750")

        self.root.minsize(850, 650)

        self.create_interface()

        self.update_interface()


    # ========================================================
    # INTERFACE
    # ========================================================

    def create_interface(self):

        title = tk.Label(
            self.root,
            text="HYBRID TRANSFORMER Q&A SYSTEM",
            font=("Arial", 20, "bold")
        )

        title.pack(pady=15)


        # ----------------------------------------------------
        # MODE
        # ----------------------------------------------------

        mode_frame = tk.Frame(self.root)

        mode_frame.pack(
            fill="x",
            padx=25,
            pady=5
        )


        tk.Label(
            mode_frame,
            text="Q&A Mode:",
            font=("Arial", 11, "bold")
        ).pack(side="left")


        self.mode_var = tk.StringVar(
            value="Closed-Domain"
        )


        self.mode_menu = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=[
                "Closed-Domain",
                "Open-Domain"
            ],
            state="readonly",
            width=25
        )

        self.mode_menu.pack(
            side="left",
            padx=10
        )

        self.mode_menu.bind(
            "<<ComboboxSelected>>",
            lambda event: self.update_interface()
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_frame = tk.Frame(self.root)

        model_frame.pack(
            fill="x",
            padx=25,
            pady=5
        )


        tk.Label(
            model_frame,
            text="Model:",
            font=("Arial", 11, "bold")
        ).pack(side="left")


        self.model_var = tk.StringVar(
            value="Extractive BERT"
        )


        self.model_menu = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[
                "Extractive BERT",
                "Generative FLAN-T5"
            ],
            state="readonly",
            width=25
        )

        self.model_menu.pack(
            side="left",
            padx=10
        )


        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        self.context_label = tk.Label(
            self.root,
            text="Context / Document:",
            font=("Arial", 11, "bold")
        )

        self.context_label.pack(
            anchor="w",
            padx=25,
            pady=(15, 5)
        )


        context_frame = tk.Frame(self.root)

        context_frame.pack(
            fill="both",
            padx=25
        )


        self.context_text = tk.Text(
            context_frame,
            height=10,
            wrap="word",
            font=("Arial", 11)
        )

        self.context_text.pack(
            side="left",
            fill="both",
            expand=True
        )


        context_scroll = ttk.Scrollbar(
            context_frame,
            command=self.context_text.yview
        )

        context_scroll.pack(
            side="right",
            fill="y"
        )


        self.context_text.config(
            yscrollcommand=context_scroll.set
        )


        # ----------------------------------------------------
        # DOCUMENT BUTTON
        # ----------------------------------------------------

        document_frame = tk.Frame(self.root)

        document_frame.pack(
            fill="x",
            padx=25,
            pady=5
        )


        self.open_button = ttk.Button(
            document_frame,
            text="Load Document",
            command=self.load_document
        )

        self.open_button.pack(
            side="left"
        )


        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        tk.Label(
            self.root,
            text="Question:",
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w",
            padx=25,
            pady=(10, 5)
        )


        self.question_text = tk.Text(
            self.root,
            height=3,
            wrap="word",
            font=("Arial", 11)
        )

        self.question_text.pack(
            fill="x",
            padx=25
        )


        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        button_frame = tk.Frame(self.root)

        button_frame.pack(
            pady=15
        )


        self.ask_button = ttk.Button(
            button_frame,
            text="ASK QUESTION",
            command=self.ask_question
        )

        self.ask_button.pack(
            side="left",
            padx=5
        )


        self.clear_button = ttk.Button(
            button_frame,
            text="CLEAR",
            command=self.clear_fields
        )

        self.clear_button.pack(
            side="left",
            padx=5
        )


        self.save_button = ttk.Button(
            button_frame,
            text="SAVE ANSWER",
            command=self.save_answer
        )

        self.save_button.pack(
            side="left",
            padx=5
        )


        self.exit_button = ttk.Button(
            button_frame,
            text="EXIT",
            command=self.root.destroy
        )

        self.exit_button.pack(
            side="left",
            padx=5
        )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        tk.Label(
            self.root,
            text="Answer:",
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w",
            padx=25,
            pady=(5, 5)
        )


        answer_frame = tk.Frame(self.root)

        answer_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=(0, 15)
        )


        self.answer_text = tk.Text(
            answer_frame,
            height=8,
            wrap="word",
            font=("Arial", 11)
        )

        self.answer_text.pack(
            side="left",
            fill="both",
            expand=True
        )


        answer_scroll = ttk.Scrollbar(
            answer_frame,
            command=self.answer_text.yview
        )

        answer_scroll.pack(
            side="right",
            fill="y"
        )


        self.answer_text.config(
            yscrollcommand=answer_scroll.set
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status_var = tk.StringVar(
            value="Ready"
        )


        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken"
        )

        status.pack(
            fill="x",
            side="bottom"
        )


    # ========================================================
    # UPDATE INTERFACE
    # ========================================================

    def update_interface(self):

        mode = self.mode_var.get()

        if mode == "Open-Domain":

            self.context_label.config(
                text="Knowledge Base:"
            )

            self.context_text.config(
                state="disabled"
            )

            self.open_button.config(
                state="disabled"
            )

            self.model_var.set(
                "Generative FLAN-T5"
            )

            self.model_menu.config(
                state="disabled"
            )

        else:

            self.context_label.config(
                text="Context / Document:"
            )

            self.context_text.config(
                state="normal"
            )

            self.open_button.config(
                state="normal"
            )

            self.model_menu.config(
                state="readonly"
            )


    # ========================================================
    # LOAD DOCUMENT
    # ========================================================

    def load_document(self):

        filename = filedialog.askopenfilename(
            title="Select a text document",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return


        try:

            with open(
                filename,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()


            self.context_text.delete(
                "1.0",
                tk.END
            )

            self.context_text.insert(
                tk.END,
                content
            )


            self.status_var.set(
                "Document loaded successfully."
            )


        except Exception as error:

            messagebox.showerror(
                "Error",
                f"Could not load document:\n{error}"
            )


    # ========================================================
    # EXTRACTIVE QA
    # ========================================================

    def extractive_answer(
        self,
        context,
        question
    ):

        inputs = qa_tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True
        )


        with torch.no_grad():

            outputs = qa_model(
                **inputs
            )


        start = torch.argmax(
            outputs.start_logits
        )

        end = torch.argmax(
            outputs.end_logits
        )


        if end < start:

            return "No suitable answer found."


        answer_tokens = inputs[
            "input_ids"
        ][0][start:end + 1]


        answer = qa_tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True
        )


        return answer


    # ========================================================
    # GENERATIVE QA
    # ========================================================

    def generative_answer(
        self,
        context,
        question
    ):

        prompt = f"""
Answer the question using only the information provided.

Context:
{context}

Question:
{question}

Answer:
"""


        inputs = gen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )


        with torch.no_grad():

            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=100
            )


        answer = gen_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        return answer


    # ========================================================
    # OPEN DOMAIN QA
    # ========================================================

    def open_domain_answer(
        self,
        question
    ):

        results = retriever.search(
            question,
            top_k=3
        )


        if not results:

            return "No relevant information was found."


        context = "\n".join(
            result["document"]
            for result in results
        )


        answer = self.generative_answer(
            context,
            question
        )


        return answer


    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(self):

        question = self.question_text.get(
            "1.0",
            tk.END
        ).strip()


        if not question:

            messagebox.showwarning(
                "Missing Question",
                "Please enter a question."
            )

            return


        self.status_var.set(
            "Processing question..."
        )

        self.root.update()


        try:

            mode = self.mode_var.get()


            # ------------------------------------------------
            # OPEN DOMAIN
            # ------------------------------------------------

            if mode == "Open-Domain":

                answer = self.open_domain_answer(
                    question
                )


            # ------------------------------------------------
            # CLOSED DOMAIN
            # ------------------------------------------------

            else:

                context = self.context_text.get(
                    "1.0",
                    tk.END
                ).strip()


                if not context:

                    messagebox.showwarning(
                        "Missing Context",
                        "Please enter or load a context."
                    )

                    self.status_var.set(
                        "Ready"
                    )

                    return


                model = self.model_var.get()


                if model == "Extractive BERT":

                    answer = self.extractive_answer(
                        context,
                        question
                    )

                else:

                    answer = self.generative_answer(
                        context,
                        question
                    )


            self.answer_text.delete(
                "1.0",
                tk.END
            )

            self.answer_text.insert(
                tk.END,
                answer
            )


            self.status_var.set(
                "Question answered successfully."
            )


        except Exception as error:

            messagebox.showerror(
                "Q&A Error",
                str(error)
            )

            self.status_var.set(
                "An error occurred."
            )


    # ========================================================
    # CLEAR
    # ========================================================

    def clear_fields(self):

        self.context_text.config(
            state="normal"
        )

        self.context_text.delete(
            "1.0",
            tk.END
        )

        self.question_text.delete(
            "1.0",
            tk.END
        )

        self.answer_text.delete(
            "1.0",
            tk.END
        )

        self.update_interface()

        self.status_var.set(
            "Ready"
        )


    # ========================================================
    # SAVE ANSWER
    # ========================================================

    def save_answer(self):

        answer = self.answer_text.get(
            "1.0",
            tk.END
        ).strip()


        if not answer:

            messagebox.showwarning(
                "No Answer",
                "There is no answer to save."
            )

            return


        filename = filedialog.asksaveasfilename(
            title="Save Answer",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt")
            ]
        )


        if not filename:

            return


        try:

            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(answer)


            messagebox.showinfo(
                "Saved",
                "Answer saved successfully."
            )


        except Exception as error:

            messagebox.showerror(
                "Error",
                f"Could not save answer:\n{error}"
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = HybridQAApp(root)

    root.mainloop()